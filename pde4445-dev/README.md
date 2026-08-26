# Picker-Bot PDE4445

### Depth-Driven Pick Sequencing with Failure-Aware Verification for Clearing Unstructured Piles of Interlocking Microelectronic Modules on an EPSON VT6-A901S

**Faseeh Mohammed** (M01088120) · MSc Robotics, Middlesex University Dubai · Module **PDE4445**
Supervisor: **Dr. Sameer** · Companion thesis (end-effector & proprioceptive): **Aman Mishra**
Platform: **EPSON VT6-A901S** 6-axis arm · **Intel RealSense D435** · **YOLOv8-OBB**

---

## The project in one glance

Tip a bucket of small hobby modules (Arduino, ESP32, LCD) onto a table and you don't get a tidy flat scatter — you get a **pile**: some parts free, some stacked, and some **hooked together through their GPIO pins**. The existing Picker-Bot assumes every part lies flat at one calibrated height, so it mis-locates anything raised, has no idea what order to clear the pile in, and never checks whether a pick worked. This project rebuilds the pipeline around a depth camera and **two cheap, honest ideas**:

- ** Topmost-first sequencing.** Clear the pile from the top down. Because the parts are near-identical in height, what matters isn't how *tall* a part is but how high its top sits *in the pile* — a single `sort()` on the depth map recovers the "what's accessible next" signal that heavier planners compute expensively.
- ** Failure-aware verification.** After each pick, re-scan and ask *did exactly one object leave?* If a pick fails or drags a tangled neighbour, **detect it and skip** — bounding the one failure mode (pin entanglement) the system can't solve.

---