package com.example.necklacedataprocessing

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.Socket
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private val piIpAddress = "10.108.233.5" // Replace with hotspot-assigned Pi IP
    private val piPort = 8765

    private lateinit var tvStatus: TextView
    private lateinit var ivNecklaceFeed: ImageView
    private lateinit var etSpeechInput: EditText
    private lateinit var btnSendSpeech: Button

    private var activeSocket: Socket? = null
    @Volatile private var isConnected = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        ivNecklaceFeed = findViewById(R.id.ivNecklaceFeed)
        etSpeechInput = findViewById(R.id.etSpeechInput)
        btnSendSpeech = findViewById(R.id.btnSendSpeech)

        connectToPiNode()

        btnSendSpeech.setOnClickListener {
            val textToSend = etSpeechInput.text.toString().trim()
            if (textToSend.isNotEmpty() && isConnected) {
                sendSpeechPayload(textToSend)
                etSpeechInput.text.clear()
            }
        }
    }

    private fun connectToPiNode() {
        thread {
            while (!isFinishing) {
                try {
                    runOnUiThread {
                        tvStatus.text = "Status: Connecting to $piIpAddress..."
                        tvStatus.setTextColor(getColor(android.R.color.holo_orange_light))
                    }

                    val socket = Socket(piIpAddress, piPort)
                    activeSocket = socket
                    isConnected = true

                    runOnUiThread {
                        tvStatus.text = "Status: Connected to Necklace Node"
                        tvStatus.setTextColor(getColor(android.R.color.holo_green_light))
                    }

                    val reader = BufferedReader(InputStreamReader(socket.getInputStream()))

                    while (isConnected) {
                        val line = reader.readLine() ?: break
                        val json = JSONObject(line)

                        if (json.optString("type") == "video_frame") {
                            val imageB64 = json.getString("image")
                            val decodedBytes = Base64.decode(imageB64, Base64.DEFAULT)
                            val bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size)

                            runOnUiThread {
                                ivNecklaceFeed.setImageBitmap(bitmap)
                            }
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                } finally {
                    isConnected = false
                    activeSocket?.close()
                    runOnUiThread {
                        tvStatus.text = "Status: Reconnecting in 3s..."
                        tvStatus.setTextColor(getColor(android.R.color.holo_red_light))
                    }
                    Thread.sleep(3000)
                }
            }
        }
    }

    private fun sendSpeechPayload(textMessage: String) {
        thread {
            try {
                activeSocket?.let { socket ->
                    val payload = JSONObject().apply {
                        put("type", "speech_output")
                        put("text", textMessage)
                        put("audio_data", "")
                        put("sample_rate", 16000)
                    }.toString() + "\n"

                    val writer = BufferedWriter(OutputStreamWriter(socket.getOutputStream()))
                    writer.write(payload)
                    writer.flush()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isConnected = false
        activeSocket?.close()
    }
}