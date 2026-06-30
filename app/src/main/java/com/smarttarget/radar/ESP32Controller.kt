package com.smarttarget.radar

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

class ESP32Controller(
    private val esp32Ip: String = "192.168.4.1",
    private val esp32Port: Int = 80
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(1, TimeUnit.SECONDS)
        .readTimeout(1, TimeUnit.SECONDS)
        .build()

    private var lastSignalTime = 0L
    private val minIntervalMs = 500L // max 2 requests per second
    private var lastSignal = ""
    private val scope = CoroutineScope(Dispatchers.IO + Job())

    fun sendSignal(signal: String) {
        if (signal == lastSignal) return

        val now = System.currentTimeMillis()
        if (now - lastSignalTime < minIntervalMs) return

        lastSignalTime = now
        lastSignal = signal

        scope.launch {
            try {
                val url = "http://$esp32Ip:$esp32Port/$signal"
                val request = Request.Builder().url(url).build()
                client.newCall(request).execute().close()
            } catch (_: Exception) {
                // ESP32 unreachable — silently ignore
            }
        }
    }
}
