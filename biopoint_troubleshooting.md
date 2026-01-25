# Sifi Labs BioPoint Sensor - Comprehensive Troubleshooting Guide

**Document Version:** 1.1  
**Last Updated:** January 2026  
**Tested On:** Ubuntu 22.04, Python 3.10

---

## Table of Contents

1. [Device Overview](#device-overview)
2. [Power & Hardware Troubleshooting](#power--hardware-troubleshooting)
3. [BLE Connection Issues](#ble-connection-issues)
4. [Python API Troubleshooting](#python-api-troubleshooting)
5. [Data Acquisition Problems](#data-acquisition-problems)
6. [Common Error Messages](#common-error-messages)
7. [Performance Optimization](#performance-optimization)
8. [FAQ](#faq)

---

## Device Overview

### BioPoint Specifications

The Sifi Labs BioPoint is a smartwatch-sized (36mm x 31mm x 11mm) wearable biosensor module featuring:

| Sensor | Specification |
|--------|---------------|
| **EMG** | 2000 Hz sampling rate |
| **ECG** | 500 Hz sampling rate |
| **PPG** | 4-channel (Blue, Green, Red, IR) @ 200Hz |
| **IMU** | 6-axis (3-axis accel, 3-axis gyro) |
| **EDA/GSR** | Electrodermal Activity sensor |
| **Temperature** | Skin temperature sensor |
| **Storage** | 4GB onboard memory |
| **Battery** | 18+ hours full-day autonomy |
| **Connectivity** | BLE (Bluetooth Low Energy) |
| **Feedback** | Visual (2 LEDs) + Vibrotactile |

---

## Power & Hardware Troubleshooting

**Important Hardware Safety Warning:**
The maximum current for the PPG LED is **50mA**. Setting the current above this value will damage the sensor. For optimal performance, we recommend a setting of **30mA** with High Sensitivity.

### Issue 1: Device Won't Power On

**Symptoms:**
- No LED indicators
- No vibration on button press
- Device appears completely unresponsive

**Solutions:**

1. **Check Battery Level**
   - Connect to charging cable for 15-30 minutes
   - Look for LED indicator changes (typically amber/red for charging, green for charged)
   - Try power-on again after charging minimum 1 hour

2. **Force Reset**
   ```
   1. Disconnect charging cable
   2. Hold power button for 10-15 seconds
   3. Release and wait 2 seconds
   4. Press power button once briefly
   ```

3. **Check Charging Port**
   - Inspect micro-USB port for debris or corrosion
   - Clean gently with dry lint-free cloth
   - Try different charging cable (verify it supports data transfer, not just charging)
   - Test charging on different USB power source (computer port better than wall adapter initially)

4. **Hardware Inspection**
   - Check for physical damage, cracks, or moisture
   - Ensure no liquid contact has occurred
   - Verify all connector pins are intact
   - Contact Sifi Labs support if hardware appears damaged

---

## BLE Connection Issues

### Issue 4: Device Not Discoverable

**Symptoms:**
- BioPoint doesn't appear in BLE scan results
- Cannot find device in phone/computer Bluetooth settings
- MAC address not visible in system

**Solutions:**

1. **Verify BLE is Enabled on Device**
   ```bash
   bluetoothctl
   > show
   ```
   - If not powered, enable with: `power on`
   - Restart Bluetooth service: `sudo systemctl restart bluetooth`

2. **Put Device in Pairing Mode**
   - Power off BioPoint completely
   - Hold power button for 3-5 seconds (until LED blinks pattern)
   - Device should be discoverable for 60-120 seconds
   - Initiate scan from host within this window

3. **Clear BLE Cache**
   ```bash
   sudo rm -rf /var/lib/bluetooth/*/
   sudo systemctl restart bluetooth
   bluetoothctl
   > scan on
   ```

4. **Reset BioPoint BLE Stack**
   - Power cycle device 2-3 times rapidly
   - Wait 10 seconds between power cycles
   - Try discovery again
   - If persistent, do factory reset (check device manual)

### Issue 5: Connection Drops Frequently

**Symptoms:**
- BLE connection established but disconnects after 5-30 seconds
- "Connection timeout" errors during data transfer
- Cannot maintain stable connection beyond brief periods

**Solutions:**

1. **Verify Signal Strength**
   ```bash
   bluetoothctl
   > info [device_mac_address]
   ```
   - Move device closer to host (within 5-10 meters for BLE)
   - BioPoint at optimal signal range: -50 to -70 dBm
   - If RSSI worse than -85 dBm, connection will be unstable

2. **Reduce Connection Interval Negotiation**
   ```python
   import sifi_bridge_py as sbp
   
   sb = sbp.SifiBridge()
   sb.connect()
   ```

3. **Check for Interference**
   - Move away from microwave ovens, WiFi, 5GHz devices
   - Bluetooth operates on 2.4GHz ISM band (same as WiFi)
   - Use WiFi band steering: WiFi on 5GHz if possible
   - Avoid placing device near metal enclosures

4. **Firmware Update**
   - Ensure BioPoint firmware is current
   - Check with Sifi Labs for latest version
   - Firmware updates may improve BLE stability
   - Use Sifi Labs official update tool (not manual installation)

5. **Host-Side Issues**
   ```python
   import time
   import sifi_bridge_py as sbp
   
   sb = sbp.SifiBridge()
   try:
       sb.connect()
   except Exception:
       time.sleep(2.0)
       sb.connect()
   ```

---

## Python API Troubleshooting

### Issue 7: Module Import Fails

**Symptoms:**
- ModuleNotFoundError: No module named 'sifi_bridge_py'
- Cannot import package in Python script

**Solutions:**

1. **Verify Installation**
   ```bash
   pip install sifi_bridge_py
   ```

2. **Check Repository Links**
   Refer to the official repositories for the latest source code and documentation:
   - **Bridge Public Repo:** https://github.com/SiFiLabs/sifi-bridge-pub
   - **Python API Repo:** https://github.com/SiFiLabs/sifi-bridge-py

3. **Python Environment**
   - Ensure you are running Python 3.10 (as recommended)
   - Check if you are in the correct virtual environment

### Issue 8: Cannot Connect via Python API

**Symptoms:**
- Script hangs at connect()
- Timeout errors when initializing SifiBridge

**Solutions:**

1. **Check Bluetooth Status Code**
   ```python
   import subprocess
   
   def _check_bluetooth_status():
       try:
           result = subprocess.run(['hciconfig'], capture_output=True, text=True)
           return "UP RUNNING" in result.stdout
       except Exception:
           return False
   ```

2. **Verify Device State**
   - Ensure device is powered on and not connected to another host
   - LED should be blinking (advertising) before script runs

---

## Data Acquisition Problems

### Issue 9: No Sensor Data Received

**Symptoms:**
- Connection successful, but get_ppg() returns empty or None
- Data stream appears dead

**Solutions:**

1. **Verify Sensor Configuration**
   ```python
   import sifi_bridge_py as sbp
   
   sb = sbp.SifiBridge()
   sb.connect()
   
   sb.configure_ppg(
       ir=0,
       green=30,
       red=0,
       blue=0,
       sens=sbp.PpgSensitivity.HIGH
   )
   
   sb.start()
   ```

2. **Check Polling Loop**
   ```python
   try:
       packet = sb.get_ppg()
       if packet and "data" in packet:
           print("Data received")
   except Exception as e:
       print(f"Read error: {e}")
   ```

---

## Common Error Messages

### "BLE Device Offline"

**Cause:** Device lost connection or powered off

**Solution:**
```python
import sifi_bridge_py as sbp
import time

sb = sbp.SifiBridge()

try:
    sb.connect()
except Exception as e:
    print(f"Error: {e}")
```

### "MTU Exchange Failed"

**Cause:** Device doesn't support requested MTU size or negotiation failed. The Bridge handles this internally, but ensure signal quality is good.

**Solution:**
- Move device closer to receiver.
- Restart the Python script to re-initiate the handshake.

### "Sensor read error"

**Cause:** Timeout during get_ppg() call or broken pipe.

**Solution:**
```python
import threading

def get_ppg_with_timeout(sb, timeout=2.0):
    result = [None]
    def worker():
        try:
            result[0] = sb.get_ppg()
        except Exception as e:
            result[0] = e
            
    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        return TimeoutError("get_ppg() timeout")
    return result[0]
```

---

## Performance Optimization

### Optimize for Real-Time Processing

Use a threaded approach to separate data fetching from processing, preventing buffer overflows.

```python
import threading
import sifi_bridge_py as sbp
import time

class SensorNode:
    def __init__(self):
        self.sb = sbp.SifiBridge()
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        
    def loop(self):
        self.sb.connect()
        self.sb.configure_ppg(green=30, sens=sbp.PpgSensitivity.HIGH)
        self.sb.start()
        
        while self.running:
            try:
                packet = self.sb.get_ppg()
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.sb.stop()
        self.sb.disconnect()
```

---

## FAQ

### Q1: What's the maximum range for BLE connection?

**A:** Standard BLE range is 10-100 meters in open space, but practical range for reliable data streaming is 5-10 meters. Walls and obstacles reduce range significantly. Use RSSI monitoring to verify signal quality.

### Q2: Can I connect multiple BioPoints simultaneously?

**A:** Yes, the BioPoint is designed for multiple simultaneous connections. However, each connection reduces available bandwidth. With 6+ devices, expect reduced sampling rates or increased latency.

### Q3: What's the typical latency for data delivery?

**A:** End-to-end latency is typically 50-200 ms depending on:
- BLE connection interval (20-40 ms typical)
- MTU size and packet count
- Data processing loop efficiency

### Q4: How do I know if firmware update is needed?

**A:** Check the SiFi Labs public repository or support channels. Major updates usually accompany new features in the Bridge API.