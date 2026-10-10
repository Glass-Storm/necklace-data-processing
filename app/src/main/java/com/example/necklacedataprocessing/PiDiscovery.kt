package com.example.necklacedataprocessing

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.Inet4Address
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Locates the Raspberry Pi node on the local network.
 *
 * Background — why discovery is needed at all:
 *   The Pi joins the phone's hotspot as a DHCP client, so its address is assigned
 *   dynamically. It cannot be hardcoded, because the subnet differs between devices
 *   and Android versions: we measured 10.92.208.x on our test handset, while other
 *   builds use 192.168.43.x or 192.168.1.x. The app therefore reads its own hotspot
 *   address at runtime and derives the subnet from it.
 *
 * Strategy (two stages, fastest first):
 *   1. Try the last address that worked, cached in SharedPreferences. ~100 ms.
 *   2. Scan the derived /24 subnet for a host that completes our handshake. ~3 s.
 *
 * Identity verification:
 *   A host is not accepted merely because a socket is listening on the port. The Pi
 *   sends a greeting on connect declaring type=hello, device=necklace-pi, a device ID
 *   and a protocol version. The probe reads that line and validates it, so an
 *   unrelated service on the same port is rejected, and a specific node can be
 *   targeted when several are present.
 *
 * The scan is bounded: 254 addresses, 20 concurrent connections, 250 ms connect
 * timeout each. That is negligible traffic on a private network and finishes in a
 * few seconds even in the worst case.
 *
 * All callbacks are delivered on a background thread, so the caller must marshal
 * UI updates onto the main thread.
 */
class PiDiscovery(
    private val context: Context,
    private val listener: Listener,
) {

    interface Listener {
        /** Progress message suitable for a status label. */
        fun onStatus(message: String)

        /**
         * A verified necklace node was found.
         *
         * @param ipAddress its address on the local network
         * @param port the TCP port its server listens on
         * @param deviceId the identifier it reported in its greeting
         */
        fun onFound(ipAddress: String, port: Int, deviceId: String)

        /** No necklace node answered on the local subnet. */
        fun onNotFound()
    }

    companion object {
        private const val TAG = "PiDiscovery"

        /** Prefs file holding the last address that worked, to skip the scan next time. */
        private const val PREFS_NAME = "pi_discovery"
        private const val KEY_LAST_IP = "last_known_ip"

        /** Connect timeout per address. A live host answers immediately. */
        private const val PROBE_TIMEOUT_MS = 250

        /** Read timeout for the greeting. Longer, since the Pi must write it first. */
        private const val HELLO_READ_TIMEOUT_MS = 1000

        /** How many addresses to probe at once. */
        private const val PROBE_THREADS = 20

        /** Host range to scan on a /24 subnet. .0 and .255 are network/broadcast. */
        private const val HOST_RANGE_START = 1
        private const val HOST_RANGE_END = 254

        /** Overall ceiling for the scan so the UI can never hang indefinitely. */
        private const val SCAN_TIMEOUT_SECONDS = 20L

        /** Values the Pi must report for us to accept it as a necklace node. */
        private const val HELLO_TYPE = "hello"
        private const val DEVICE_NAME = "necklace-pi"
    }

    /** A verified node: its address plus the identifier it reported. */
    private data class Node(val ipAddress: String, val deviceId: String)

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val cancelled = AtomicBoolean(false)

    /** Stop an in-flight scan. Safe to call from any thread. */
    fun cancel() {
        cancelled.set(true)
    }

    /**
     * Run discovery on a background thread.
     *
     * @param port the TCP port the Pi's server listens on
     */
    fun start(port: Int) {
        cancelled.set(false)

        Thread({
            // --- Stage 1: last known address -------------------------------------
            val lastIp = prefs.getString(KEY_LAST_IP, null)
            if (lastIp != null && !cancelled.get()) {
                Log.d(TAG, "Trying last known address $lastIp")
                listener.onStatus("Checking last known Pi at $lastIp...")

                val cached = verifyNode(lastIp, port)
                if (cached != null) {
                    Log.d(TAG, "Last known address valid (device ${cached.deviceId})")
                    listener.onFound(cached.ipAddress, port, cached.deviceId)
                    return@Thread
                }
                Log.d(TAG, "Last known address no longer answers")
            }

            if (cancelled.get()) return@Thread

            // --- Stage 2: subnet scan --------------------------------------------
            val localIp = findHotspotAddress()
            if (localIp == null) {
                Log.w(TAG, "No local address found — is the hotspot on?")
                listener.onNotFound()
                return@Thread
            }

            val subnet = localIp.substringBeforeLast('.')
            Log.d(TAG, "Local address $localIp, scanning $subnet.0/24")

            val found = scanSubnet(subnet, port, localIp)

            if (cancelled.get()) return@Thread

            if (found != null) {
                prefs.edit().putString(KEY_LAST_IP, found.ipAddress).apply()
                listener.onFound(found.ipAddress, port, found.deviceId)
            } else {
                listener.onNotFound()
            }
        }, "pi-discovery").start()
    }

    /** Forget the cached address, forcing a full scan next time. */
    fun clearCache() {
        prefs.edit().remove(KEY_LAST_IP).apply()
    }

    // ---------------------------------------------------------------------------
    // Subnet scanning
    // ---------------------------------------------------------------------------

    /**
     * Probe every host address on [subnet] for a necklace node.
     *
     * @return the first verified node, or null if none answered.
     */
    private fun scanSubnet(subnet: String, port: Int, skipAddress: String): Node? {
        listener.onStatus("Scanning $subnet.0/24 for the Pi...")
        Log.d(TAG, "Scanning $subnet.$HOST_RANGE_START-$HOST_RANGE_END on port $port")

        val started = System.currentTimeMillis()
        val probed = AtomicInteger(0)

        val pool = Executors.newFixedThreadPool(PROBE_THREADS)
        val result = AtomicReference<Node?>(null)

        try {
            for (host in HOST_RANGE_START..HOST_RANGE_END) {
                if (cancelled.get() || result.get() != null) break

                val candidate = "$subnet.$host"
                // The phone's own hotspot address is not the Pi.
                if (candidate == skipAddress) continue

                pool.submit {
                    if (cancelled.get() || result.get() != null) return@submit

                    probed.incrementAndGet()
                    val node = verifyNode(candidate, port)
                    if (node != null) {
                        Log.d(TAG, "Found necklace node ${node.deviceId} at $candidate")
                        // compareAndSet ensures only the first responder wins.
                        result.compareAndSet(null, node)
                    }
                }
            }

            // Wait for the probes to finish, but never longer than the ceiling.
            pool.shutdown()
            pool.awaitTermination(SCAN_TIMEOUT_SECONDS, TimeUnit.SECONDS)

            val elapsed = System.currentTimeMillis() - started
            Log.d(TAG, "Scan complete: probed ${probed.get()} addresses in ${elapsed}ms")
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        } finally {
            pool.shutdownNow()
        }

        return result.get()
    }

    /**
     * Confirm a candidate host is a necklace node by reading its greeting.
     *
     * The Pi sends a newline-delimited JSON message immediately on connect:
     *   {"type":"hello","device":"necklace-pi","device_id":"necklace-01","protocol":1}
     *
     * A plain port check would accept any listener. Validating the greeting proves
     * the host speaks our protocol, which also rules out an unrelated service that
     * happens to occupy the same port.
     *
     * The probe disconnects as soon as the line is read. The Pi sends the greeting
     * before opening its camera, so a scan does not start a video session.
     *
     * @return the node if verified, or null when the host is not ours.
     */
    private fun verifyNode(ip: String, port: Int): Node? {
        return try {
            Socket().use { socket ->
                socket.connect(InetSocketAddress(ip, port), PROBE_TIMEOUT_MS)
                // Allow longer than the connect timeout: the Pi must accept the
                // connection and write the greeting before we can read it.
                socket.soTimeout = HELLO_READ_TIMEOUT_MS

                val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
                val line = reader.readLine() ?: return null

                val json = JSONObject(line)
                if (json.optString("type") != HELLO_TYPE) return null
                if (json.optString("device") != DEVICE_NAME) return null

                val deviceId = json.optString("device_id").ifBlank { "unknown" }
                Node(ip, deviceId)
            }
        } catch (e: Exception) {
            // Not our protocol, not reachable, or the greeting never arrived.
            null
        }
    }

    // ---------------------------------------------------------------------------
    // Local address discovery
    // ---------------------------------------------------------------------------

    /**
     * Return the phone's own IPv4 address on the Wi-Fi interface.
     *
     * When the phone hosts a hotspot, its wlan0 interface holds the access point
     * address (for example 10.92.208.1). Enumerating interfaces is used rather than
     * WifiManager.getDhcpInfo() because the latter is deprecated from API 31 and
     * reports the DHCP lease rather than the AP address on some devices.
     *
     * @return the address, or null when no Wi-Fi interface is up.
     */
    private fun findHotspotAddress(): String? {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces() ?: return null

            // Candidate addresses, in preference order. Interface naming varies between
            // OEMs, so we log everything and score rather than matching a fixed list.
            var best: String? = null

            for (iface in interfaces) {
                if (!iface.isUp || iface.isLoopback) continue

                val name = iface.name.lowercase()

                for (address in iface.inetAddresses) {
                    if (address !is Inet4Address || address.isLoopbackAddress) continue

                    val ip = address.hostAddress ?: continue
                    Log.d(TAG, "Interface $name -> $ip")

                    // Private ranges only: 10.x, 172.16-31.x, 192.168.x
                    if (!isPrivateIpv4(ip)) continue

                    // Prefer interfaces that look like a Wi-Fi/AP interface.
                    val looksWireless = name.startsWith("wlan") ||
                        name.startsWith("ap") ||
                        name.startsWith("softap") ||
                        name.startsWith("swlan")

                    if (looksWireless) {
                        Log.d(TAG, "Selected wireless interface $name ($ip)")
                        return ip
                    }

                    // Remember a private address as a fallback if nothing wireless matches.
                    if (best == null) best = ip
                }
            }

            if (best != null) {
                Log.w(TAG, "No wireless interface matched; falling back to $best")
            } else {
                Log.w(TAG, "No private IPv4 address found on any interface")
            }
            return best
        } catch (e: Exception) {
            Log.e(TAG, "Interface enumeration failed: ${e.message}")
        }
        return null
    }

    /** True for RFC 1918 private IPv4 addresses. */
    private fun isPrivateIpv4(ip: String): Boolean {
        val parts = ip.split('.')
        if (parts.size != 4) return false
        val a = parts[0].toIntOrNull() ?: return false
        val b = parts[1].toIntOrNull() ?: return false

        return when (a) {
            10 -> true
            172 -> b in 16..31
            192 -> b == 168
            else -> false
        }
    }
}
