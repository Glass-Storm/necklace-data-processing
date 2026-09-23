package com.example.necklacedataprocessing

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.InetSocketAddress
import java.net.Socket
import kotlin.concurrent.thread

/**
 * MainActivity — Android Hub Client
 *
 * Acts as the TCP client in the Pi <-> Phone communication link.
 * - Receives Base64-encoded JPEG frames from the Raspberry Pi and renders them live.
 * - Sends speech payloads back to the Pi for playback on the necklace speaker.
 *
 * Protocol: newline-delimited JSON over TCP (port 8765)
 *
 * Connection flow:
 *   1. User taps "Scan for Pi" — the app locates the Pi on the hotspot subnet.
 *   2. If found, the address is shown and "Connect" becomes available.
 *   3. User taps "Connect" — the TCP stream starts and frames render live.
 *
 * The Pi's address cannot be hardcoded: it is assigned by the phone's DHCP server and
 * the subnet varies by device (10.92.208.x on our test handset, 192.168.43.x on others).
 */
class MainActivity : AppCompatActivity(), PiDiscovery.Listener, TtsAudioEncoder.Callback {

    companion object {
        private const val TAG = "MainActivity"
        private const val PI_PORT = 8765

        /** Connect/read timeout for the video socket. */
        private const val SOCKET_TIMEOUT_MS = 5000

        /** Delay before retrying a dropped connection. */
        private const val RECONNECT_DELAY_MS = 3000L
    }

    // --- UI ---
    private lateinit var tvStatus: TextView
    private lateinit var ivNecklaceFeed: ImageView
    private lateinit var etSpeechInput: EditText
    private lateinit var btnSendSpeech: Button
    private lateinit var btnScan: Button
    private lateinit var btnConnect: Button

    // --- Discovery ---
    private var discovery: PiDiscovery? = null

    // --- Text to speech ---
    private var ttsEncoder: TtsAudioEncoder? = null

    /**
     * The text currently being synthesised.
     *
     * Sent alongside the audio purely so the Pi's log shows what was spoken —
     * playback uses the audio only. A single field is sufficient because
     * synthesis uses QUEUE_FLUSH, so only the latest utterance matters.
     */
    @Volatile
    private var pendingSpeechText: String = ""

    /** Address reported by discovery. Null until a Pi is found. */
    @Volatile
    private var discoveredIp: String? = null

    /** Identifier the Pi reported during the handshake. */
    @Volatile
    private var discoveredDeviceId: String? = null

    // --- Connection state ---
    private var activeSocket: Socket? = null

    @Volatile
    private var isConnected = false

    /** Set true once the user taps Connect, so the reconnect loop may run. */
    @Volatile
    private var shouldBeConnected = false

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        ivNecklaceFeed = findViewById(R.id.ivNecklaceFeed)
        etSpeechInput = findViewById(R.id.etSpeechInput)
        btnSendSpeech = findViewById(R.id.btnSendSpeech)
        btnScan = findViewById(R.id.btnScan)
        btnConnect = findViewById(R.id.btnConnect)

        btnScan.setOnClickListener { startDiscovery() }
        btnConnect.setOnClickListener { connectToPi() }

        // Prepare the speech engine now so it is ready by the time the user types.
        ttsEncoder = TtsAudioEncoder(this).also { it.initialize() }

        btnSendSpeech.setOnClickListener {
            val textToSend = etSpeechInput.text.toString().trim()
            // Guard clause: only synthesise when there is text AND the socket is alive.
            if (textToSend.isEmpty()) return@setOnClickListener
            if (!isConnected) {
                setStatus("Not connected to the Pi.", StatusColour.ERROR)
                return@setOnClickListener
            }

            // Synthesis is asynchronous; the result arrives in onEncoded below.
            setStatus("Synthesising speech...", StatusColour.WORKING)
            pendingSpeechText = textToSend
            ttsEncoder?.speak(textToSend, this)
            etSpeechInput.text.clear()
        }

        setStatus("Tap \"Scan for Pi\" to begin.", StatusColour.IDLE)
    }

    override fun onDestroy() {
        super.onDestroy()
        // Stop discovery and tear down the socket so nothing leaks.
        discovery?.cancel()
        ttsEncoder?.shutdown()
        ttsEncoder = null
        shouldBeConnected = false
        isConnected = false
        try {
            activeSocket?.close()
        } catch (_: Exception) {
        }
    }

    // -----------------------------------------------------------------------
    // Discovery
    // -----------------------------------------------------------------------

    private fun startDiscovery() {
        discovery?.cancel()
        discovery = PiDiscovery(this, this)

        discoveredIp = null
        discoveredDeviceId = null
        btnConnect.isEnabled = false
        btnScan.isEnabled = false

        discovery?.start(PI_PORT)
    }

    /** Progress updates from the discovery thread — marshal onto the UI thread. */
    override fun onStatus(message: String) {
        setStatus(message, StatusColour.WORKING)
    }

    /** A verified Pi answered on the subnet. Enable Connect, but wait for the user. */
    override fun onFound(ipAddress: String, port: Int, deviceId: String) {
        discoveredIp = ipAddress
        discoveredDeviceId = deviceId
        runOnUiThread {
            btnScan.isEnabled = true
            btnConnect.isEnabled = true
            setStatus("Found $deviceId at $ipAddress — tap Connect.", StatusColour.OK)
        }
    }

    /** Nothing answered. Let the user retry. */
    override fun onNotFound() {
        runOnUiThread {
            btnScan.isEnabled = true
            btnConnect.isEnabled = false
            setStatus("No Pi found. Check the hotspot and tap Scan again.", StatusColour.ERROR)
        }
    }

    // -----------------------------------------------------------------------
    // Connection
    // -----------------------------------------------------------------------

    private fun connectToPi() {
        val ip = discoveredIp ?: run {
            setStatus("No Pi address yet — scan first.", StatusColour.ERROR)
            return
        }

        // Prevent a second connect loop if the button is tapped twice.
        if (shouldBeConnected) return
        shouldBeConnected = true

        btnConnect.isEnabled = false
        btnScan.isEnabled = false

        connectLoop(ip)
    }

    /**
     * Maintain the TCP connection, reconnecting whenever it drops.
     *
     * Runs on a background thread: network I/O on the main thread would freeze the UI.
     */
    private fun connectLoop(ip: String) {
        thread {
            while (shouldBeConnected && !isFinishing) {
                try {
                    setStatus("Connecting to $ip:$PI_PORT...", StatusColour.WORKING)

                    // Connect with a timeout so an unreachable Pi fails fast instead of hanging.
                    val socket = Socket()
                    socket.connect(InetSocketAddress(ip, PI_PORT), SOCKET_TIMEOUT_MS)
                    // Without a read timeout, a silently dead Pi would block readLine() forever.
                    socket.soTimeout = SOCKET_TIMEOUT_MS

                    activeSocket = socket
                    isConnected = true
                    setStatus("Connected to Necklace Node", StatusColour.OK)

                    // Wrap the socket in a BufferedReader so we can read newline-delimited
                    // JSON lines. The Pi appends "\n" to every payload, which terminates
                    // each readLine() call.
                    val reader = BufferedReader(InputStreamReader(socket.getInputStream()))

                    while (isConnected && shouldBeConnected) {
                        // Blocks until a full line arrives. Returns null if the Pi closed the stream.
                        val line = reader.readLine() ?: break
                        val json = JSONObject(line)

                        when (json.optString("type")) {
                            // The Pi greets us on connect. Discovery already validated it,
                            // so here we just log the identity and protocol version.
                            "hello" -> {
                                val device = json.optString("device_id", "unknown")
                                val protocol = json.optInt("protocol", 0)
                                Log.d(TAG, "Pi handshake: device=$device protocol=$protocol")
                            }

                            // Reverse the Pi's pipeline: Base64 text -> binary JPEG -> Bitmap.
                            "video_frame" -> {
                                val imageB64 = json.getString("image")
                                val decodedBytes = Base64.decode(imageB64, Base64.DEFAULT)
                                val bitmap = BitmapFactory.decodeByteArray(
                                    decodedBytes, 0, decodedBytes.size,
                                )

                                // Render on the UI thread — Bitmaps cannot be drawn off-thread.
                                runOnUiThread { ivNecklaceFeed.setImageBitmap(bitmap) }
                            }

                            // Unknown types are ignored so the protocol can be extended
                            // without breaking older clients.
                            else -> Unit
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Connection error: ${e.message}")
                } finally {
                    // Always runs, even on crash — releases the socket and resets state.
                    isConnected = false
                    try {
                        activeSocket?.close()
                    } catch (_: Exception) {
                    }
                    activeSocket = null
                }

                if (shouldBeConnected && !isFinishing) {
                    setStatus("Connection lost. Retrying in 3s...", StatusColour.ERROR)
                    // Throttle retries to avoid a busy-wait loop hammering the CPU.
                    Thread.sleep(RECONNECT_DELAY_MS)
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Speech output (TTS -> PCM -> Pi speaker)
    // -----------------------------------------------------------------------

    /**
     * Called on a background thread once synthesis has produced PCM samples.
     *
     * The samples are base64-encoded 16-bit little-endian mono, which is exactly
     * what the Pi's `sounddevice.play()` expects.
     */
    override fun onEncoded(audioBase64: String, sampleRate: Int) {
        sendAudioPayload(audioBase64, sampleRate)
    }

    override fun onError(reason: String) {
        setStatus(reason, StatusColour.ERROR)
    }

    /**
     * Transmit PCM audio to the Pi for playback on its speaker.
     *
     * Runs on its own thread: writing a large base64 payload to the socket can
     * block if the network buffer is full, which must not happen on the UI thread.
     */
    private fun sendAudioPayload(audioBase64: String, sampleRate: Int) {
        thread {
            try {
                // Safe call: skip entirely if the socket is null (not connected).
                activeSocket?.let { socket ->
                    val payload = JSONObject().apply {
                        put("type", "speech_output")
                        // For the Pi's log only — playback uses the audio. Sending the
                        // text keeps the Pi's console readable during debugging.
                        put("text", pendingSpeechText)
                        put("audio_data", audioBase64)
                        put("sample_rate", sampleRate)
                    }.toString() + "\n" // Trailing newline required by the Pi's readline()

                    val writer = BufferedWriter(OutputStreamWriter(socket.getOutputStream()))
                    writer.write(payload)
                    writer.flush() // Force send immediately instead of buffering

                    Log.d(TAG, "Sent ${audioBase64.length} base64 chars at $sampleRate Hz")
                    setStatus("Audio sent to the Pi.", StatusColour.OK)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Send failed: ${e.message}")
                setStatus("Could not send audio: ${e.message}", StatusColour.ERROR)
            }
        }
    }

    // -----------------------------------------------------------------------
    // UI helpers
    // -----------------------------------------------------------------------

    private enum class StatusColour { IDLE, WORKING, OK, ERROR }

    private fun setStatus(message: String, colour: StatusColour) {
        runOnUiThread {
            tvStatus.text = message
            tvStatus.setTextColor(
                when (colour) {
                    StatusColour.IDLE -> ContextCompat.getColor(this, android.R.color.darker_gray)
                    StatusColour.WORKING -> ContextCompat.getColor(this, android.R.color.holo_orange_light)
                    StatusColour.OK -> ContextCompat.getColor(this, android.R.color.holo_green_light)
                    StatusColour.ERROR -> ContextCompat.getColor(this, android.R.color.holo_red_light)
                },
            )
        }
    }
}
