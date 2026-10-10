package com.example.necklacedataprocessing

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Base64
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * Converts text into raw PCM audio the Pi can play.
 *
 * Why this class exists:
 *   The Pi's speaker is driven by `sounddevice`, which expects a base64-encoded
 *   array of 16-bit little-endian PCM samples. Android's TextToSpeech cannot hand
 *   us samples directly, so we synthesise to a WAV file and extract the PCM from
 *   it. That is the only supported route on Android.
 *
 * Pipeline:
 *   text -> TextToSpeech.synthesizeToFile() -> WAV file
 *        -> parse RIFF chunks -> extract PCM
 *        -> downmix to mono -> base64 -> send to the Pi
 *
 * Why downmix to mono:
 *   The Pi calls `sounddevice.play(samples, samplerate=...)` with a one-dimensional
 *   array, which it interprets as mono. If we sent interleaved stereo it would play
 *   at double speed. Averaging the channels on the phone keeps the Pi's code simple
 *   and avoids adding a channel-count field to the protocol.
 *
 * Threading:
 *   TextToSpeech initialises asynchronously and reports completion on a binder
 *   thread, so [speak] may invoke its callback from a non-UI thread. Callers must
 *   marshal any UI work themselves.
 */
class TtsAudioEncoder(private val context: Context) {

    interface Callback {
        /** Synthesis succeeded. [audioBase64] is 16-bit mono PCM, little-endian. */
        fun onEncoded(audioBase64: String, sampleRate: Int)

        /** Synthesis failed, or the engine is not ready. */
        fun onError(reason: String)
    }

    companion object {
        private const val TAG = "TtsAudioEncoder"

        /** The Pi decodes with dtype=int16, so anything else would be misinterpreted. */
        private const val REQUIRED_BITS_PER_SAMPLE = 16

        /**
         * Guard against a very long utterance producing a payload that would take
         * seconds to transmit. Roughly a paragraph.
         */
        private const val MAX_TEXT_LENGTH = 300

        /** Prefix for the temporary WAV files written to the cache directory. */
        private const val TTS_FILE_PREFIX = "tts_"
    }

    /** Parsed contents of a WAV file. */
    private data class WavData(
        val pcm: ByteArray,
        val sampleRate: Int,
        val channels: Int,
        val bitsPerSample: Int,
    )

    private var tts: TextToSpeech? = null

    @Volatile
    private var ready = false

    /** Maps an utterance ID to the callback awaiting its result. */
    private val pending = ConcurrentHashMap<String, Callback>()

    /**
     * Start the TTS engine. Safe to call more than once.
     *
     * Initialisation is asynchronous: [ready] flips to true once the engine has
     * loaded, which typically takes a few hundred milliseconds.
     */
    fun initialize() {
        if (tts != null) return

        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                ready = true
                Log.d(TAG, "TTS engine ready")
            } else {
                ready = false
                Log.e(TAG, "TTS initialisation failed (status $status)")
            }
        }.apply {
            // English is the language the hearing user receives output in.
            val result = setLanguage(Locale.ENGLISH)
            if (result == TextToSpeech.LANG_MISSING_DATA ||
                result == TextToSpeech.LANG_NOT_SUPPORTED
            ) {
                Log.w(TAG, "English TTS data unavailable on this device")
            }

            setOnUtteranceProgressListener(object : UtteranceProgressListener() {

                override fun onStart(utteranceId: String?) = Unit

                /** Synthesis finished — the WAV file is ready to read. */
                override fun onDone(utteranceId: String?) {
                    val id = utteranceId ?: return
                    val callback = pending.remove(id) ?: return
                    encodeFile(id, callback)
                }

                /** Synthesis failed. */
                override fun onError(utteranceId: String?) {
                    val id = utteranceId ?: return
                    pending.remove(id)?.onError("Speech synthesis failed.")
                }

                @Deprecated("Superseded by onError(String, int) on API 21+")
                override fun onError(utteranceId: String?, errorCode: Int) {
                    val id = utteranceId ?: return
                    pending.remove(id)?.onError("Speech synthesis failed (code $errorCode).")
                }
            })
        }
    }

    /**
     * Synthesise [text] and return the result through [callback].
     *
     * @return true if synthesis was started. The result arrives asynchronously.
     */
    fun speak(text: String, callback: Callback): Boolean {
        val engine = tts
        if (engine == null || !ready) {
            callback.onError("Speech engine is still starting up.")
            return false
        }

        if (text.isBlank()) {
            callback.onError("Nothing to speak.")
            return false
        }

        if (text.length > MAX_TEXT_LENGTH) {
            callback.onError("Text is too long (max $MAX_TEXT_LENGTH characters).")
            return false
        }

        // A unique ID lets the progress listener match the result to this request.
        val utteranceId = UUID.randomUUID().toString()
        val file = File(context.cacheDir, "$TTS_FILE_PREFIX$utteranceId.wav")

        pending[utteranceId] = callback

        // QUEUE_FLUSH: a new utterance replaces anything still speaking, which is the
        // right behaviour for a conversation aid where only the latest message matters.
        val result = engine.synthesizeToFile(text, Bundle(), file, utteranceId)

        if (result != TextToSpeech.SUCCESS) {
            pending.remove(utteranceId)
            callback.onError("Could not start speech synthesis.")
            return false
        }

        return true
    }

    /** Release the engine and delete any leftover temporary files. */
    fun shutdown() {
        pending.clear()
        tts?.stop()
        tts?.shutdown()
        tts = null
        ready = false

        // Remove any WAV files left behind by an interrupted synthesis.
        context.cacheDir.listFiles { f -> f.name.startsWith(TTS_FILE_PREFIX) }
            ?.forEach { it.delete() }
    }

    // ---------------------------------------------------------------------------
    // WAV handling
    // ---------------------------------------------------------------------------

    /** Read the synthesised file, extract PCM, and hand it to the callback. */
    private fun encodeFile(utteranceId: String, callback: Callback) {
        val file = File(context.cacheDir, "$TTS_FILE_PREFIX$utteranceId.wav")

        try {
            if (!file.exists()) {
                callback.onError("Synthesis produced no file.")
                return
            }

            val wav = parseWav(file.readBytes())
            if (wav == null) {
                callback.onError("Could not parse the synthesised audio.")
                return
            }

            if (wav.bitsPerSample != REQUIRED_BITS_PER_SAMPLE) {
                callback.onError("Unexpected sample format (${wav.bitsPerSample}-bit).")
                return
            }

            val mono = downmixToMono(wav.pcm, wav.channels)
            if (mono.isEmpty()) {
                callback.onError("Synthesised audio was empty.")
                return
            }

            val base64 = Base64.encodeToString(mono, Base64.NO_WRAP)
            Log.d(TAG, "Encoded ${mono.size} bytes of PCM at ${wav.sampleRate} Hz")
            callback.onEncoded(base64, wav.sampleRate)
        } catch (e: Exception) {
            Log.e(TAG, "Encoding failed: ${e.message}")
            callback.onError("Could not read the synthesised audio.")
        } finally {
            // The file has served its purpose; keep the cache directory clean.
            file.delete()
        }
    }

    /**
     * Parse a RIFF/WAVE file and return its PCM payload.
     *
     * Chunks are walked rather than read at fixed offsets, because a WAV file may
     * contain additional chunks (LIST, fact, and so on) before the data chunk, and
     * their presence shifts every subsequent offset.
     */
    private fun parseWav(bytes: ByteArray): WavData? {
        if (bytes.size < 44) return null

        // RIFF header: "RIFF" <size> "WAVE"
        if (ascii(bytes, 0, 4) != "RIFF") return null
        if (ascii(bytes, 8, 4) != "WAVE") return null

        var offset = 12
        var sampleRate = 0
        var channels = 0
        var bitsPerSample = 0
        var pcm: ByteArray? = null

        while (offset + 8 <= bytes.size) {
            val chunkId = ascii(bytes, offset, 4)
            val chunkSize = intLE(bytes, offset + 4)
            val dataStart = offset + 8

            if (chunkSize < 0) return null

            when (chunkId) {
                "fmt " -> {
                    // fmt layout: audioFormat(2) channels(2) sampleRate(4)
                    //             byteRate(4) blockAlign(2) bitsPerSample(2)
                    if (dataStart + 16 > bytes.size) return null
                    channels = shortLE(bytes, dataStart + 2)
                    sampleRate = intLE(bytes, dataStart + 4)
                    bitsPerSample = shortLE(bytes, dataStart + 14)
                }

                "data" -> {
                    val end = minOf(dataStart + chunkSize, bytes.size)
                    if (end > dataStart) {
                        pcm = bytes.copyOfRange(dataStart, end)
                    }
                }
            }

            // Chunks are word-aligned: an odd size is followed by a pad byte.
            offset = dataStart + chunkSize + (chunkSize % 2)
        }

        if (pcm == null || sampleRate <= 0 || channels <= 0) return null
        return WavData(pcm, sampleRate, channels, bitsPerSample)
    }

    /**
     * Average interleaved channels into a single mono stream.
     *
     * The Pi plays a one-dimensional sample array, so stereo input would be
     * interpreted as mono and play at twice the intended rate.
     */
    private fun downmixToMono(pcm: ByteArray, channels: Int): ByteArray {
        if (channels <= 1) return pcm

        val frameCount = pcm.size / 2 / channels
        if (frameCount <= 0) return ByteArray(0)

        val source = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN)
        val out = ByteBuffer.allocate(frameCount * 2).order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until frameCount) {
            var sum = 0
            for (c in 0 until channels) {
                if (source.remaining() < 2) break
                sum += source.short.toInt()
            }
            out.putShort((sum / channels).toShort())
        }

        return out.array()
    }

    // ---------------------------------------------------------------------------
    // Little-endian readers
    // ---------------------------------------------------------------------------

    private fun ascii(bytes: ByteArray, offset: Int, length: Int): String =
        String(bytes, offset, length, Charsets.US_ASCII)

    private fun intLE(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xFF) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 3].toInt() and 0xFF) shl 24)

    private fun shortLE(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xFF) or ((bytes[offset + 1].toInt() and 0xFF) shl 8)
}
