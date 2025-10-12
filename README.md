---
marp: true
paginate: true
theme: default
---

# ♻️ AI-Powered Disposal Sorter
_A Smart Recycling System Using Image Classification_

---

## 🌍 Problem Statement
- Recycling systems are inefficient due to **human sorting errors**.  
- Contamination in bins reduces recyclability.  
- Most public bins **lack feedback** or smart sorting capabilities.  

---

## 💡 Proposed Solution
- An **AI-powered disposal sorter** using **image recognition**.  
- Detects item type (plastic, glass, paper, metal, organic).  
- Automatically sorts into the correct bin using a **mechanical system**.  

---

## ⚙️ System Design Overview
1. **Camera Module** – captures image of the waste item.  
2. **Edge AI Device** – runs trained model for waste classification.  
3. **Control Unit** – activates servo or motor for bin routing.  
4. **Cloud Dashboard** – monitors usage data and recycling stats.  

---

## 🧠 AI Model
- Model: **Convolutional Neural Network (CNN)**  
- Dataset: labeled waste images (plastic, glass, metal, paper, organic)  
- Trained using **TensorFlow Lite** for edge deployment  

---

## 🧩 Edge Implementation
- Runs on **Raspberry Pi** or **NVIDIA Jetson Nano**  
- Uses **OpenCV** for real-time image processing  
- Local classification ensures **fast, offline decisions**  

---

## 🧰 Hardware Components
- Raspberry Pi 4  
- USB Camera Module  
- Servo Motors (for bin gates)  
- Ultrasonic Sensor (detect item presence)  
- Power Supply  

---

## 🧾 Expected Outcomes
✅ Accurate waste sorting (>90% accuracy)  
✅ Reduced contamination in recycling streams  
✅ Real-time data collection for waste analytics  

---

## 🚀 Future Improvements
- Add **composting detection** (biodegradable recognition)  
- Integrate with **IoT platforms** for city-wide data tracking  
- Use **solar power** for sustainability  

---

## 👥 Team & Roles
| Name | Role |
|------|------|
| You | AI Developer / Project Lead |
| Partner A | Hardware Design |
| Partner B | Data Collection & Training |

---

## 🏁 Conclusion
> “AI-powered recycling isn’t just smarter — it’s cleaner, faster, and more sustainable.”

---

## 📸 Demo (Optional)
![Prototype Image](images/prototype.jpg)
*(Insert an image of your setup or concept sketch here)*

---

## 🔗 References
- TensorFlow Lite Documentation  
- OpenCV Library  
- Kaggle Waste Classification Dataset  

---

**Thank you!**
