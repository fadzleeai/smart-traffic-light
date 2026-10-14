# AI-Powered Traffic Light Control System (NextLogic)

An edge-deployed, smart traffic management system designed for a 3-way T-junction[cite: 1]. It replaces fixed timers with real-time computer vision and Monte Carlo simulation to choose the best green-light timing for each road[cite: 1].

> **Hardware Setup:** The entire system runs on a single **Raspberry Pi 5**, which handles 3 camera feeds, AI detection, simulation, and all 3 traffic light signals directly.

---

## 📌 Problem & Solution

* **The Problem:** Traditional traffic lights run on fixed timers, causing cars to wait at empty roads, wasting fuel, and causing congestion[cite: 1].
* **Our Solution:** A single Raspberry Pi 5 looks at the junction with 3 cameras, counts waiting cars, simulates the best timing, and switches the physical LED traffic lights[cite: 1].
* **Goal:** Reduce waiting time by up to ~40% and cut unnecessary vehicle idling[cite: 1].

---

## ⚙️ How It Works (4 Steps)

[3 Cameras] ──▶ [YOLOv8 Detection] ──▶ [Monte Carlo Simulation] ──▶ [GPIO / Traffic Lights]

1. **Camera Feed:** 3 cameras point at the North, East, and West approaches of the junction[cite: 1].
2. **Vehicle Counting:** An AI model (**YOLOv8n** running on ONNX Runtime at $320\times320$) counts how many cars are waiting in each lane[cite: 1].
3. **Simulation:** The **Monte Carlo algorithm** simulates hundreds of possibilities (for example, testing a 10s/35s/15s split vs a 25s/10s/25s split) to find which timing clears the most cars[cite: 1].
4. **Light Control:** The Raspberry Pi 5 sends signals through its GPIO pins to turn the physical Red, Yellow, and Green LEDs on and off[cite: 1].

---

## 🛠️ Hardware & Tools

* **Main Board:** Raspberry Pi 5
* **Cameras:** 3× Cameras (covering all 3 approaches)[cite: 1]
* **Signals:** 3× Sets of Red, Yellow, Green LEDs wired to the Pi's GPIO pins[cite: 1]
* **AI Model:** YOLOv8n (ONNX format)[cite: 1]
* **Programming Language:** Python

---

## 👥 The Team (NextLogic)

* **Mentor:** Ts. Dr. Norulhusna Binti Ahmad[cite: 1]
* **Project Leader:** Fadzlee Adam Bin Mohd Nazlee[cite: 1]
* **Team Members:**
  * Ahmad Haziq Bin Ahmad Khairuman[cite: 1]
  * Muhammad Haziem Azfar Bin Mohd Ransawon[cite: 1]
  * Shirlyn Siew Yu Jye[cite: 1]
  * Siti Noor Balqis Binti Yuslim[cite: 1]
