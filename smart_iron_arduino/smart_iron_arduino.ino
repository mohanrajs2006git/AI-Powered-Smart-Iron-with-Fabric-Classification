/*
  Smart Iron — Arduino Sensor Node
  Sends Temperature, Motion Variation, Static Time over Serial.
  Flask server reads this via pyserial.

  Wiring:
    LM35  → GPIO34  (ADC pin)
    MPU6050 or any IMU → I2C (optional, simulated here)

  Serial output format (one line per second):
    SENSOR,<temp_c>,<motion_var>,<static_s>
*/

#define LM35_PIN 34

// ── Simulated motion/static (replace with real IMU reads) ─────────
// If you have an MPU6050, read accel variance here instead.
float  motionBuffer[10];
int    mIdx = 0;
unsigned long lastMoveTime = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);       // ESP32: 12-bit ADC
  for (int i = 0; i < 10; i++) motionBuffer[i] = 0.02;
}

void loop() {
  // ── Temperature ───────────────────────────────────────────────
  int   adcValue    = analogRead(LM35_PIN);
  float voltage     = adcValue * (3.3f / 4095.0f);
  float temperature = voltage * 100.0f;           // LM35: 10 mV/°C

  // ── Motion Variation (simulated — replace with IMU) ───────────
  // Real: read MPU6050 accel magnitude, push to ring buffer, compute std-dev
  float fakeMotion  = 0.015f + (analogRead(35) % 100) * 0.0003f; // remove when using real IMU
  motionBuffer[mIdx++ % 10] = fakeMotion;
  float motionVar   = computeVariance(motionBuffer, 10);

  // ── Static Time ───────────────────────────────────────────────
  bool  moving      = (motionVar > 0.0005f);
  if (moving) lastMoveTime = millis();
  int   staticSec   = (millis() - lastMoveTime) / 1000;
  staticSec         = constrain(staticSec, 1, 45);

  // ── Clamp to model input ranges ───────────────────────────────
  temperature = constrain(temperature, 60.0f, 250.0f);
  motionVar   = constrain(motionVar,   0.001f, 0.05f);

  // ── Send to Flask ─────────────────────────────────────────────
  Serial.print("SENSOR,");
  Serial.print(temperature, 2);
  Serial.print(",");
  Serial.print(motionVar, 4);
  Serial.print(",");
  Serial.println(staticSec);

  delay(1000);    // 1 reading per second
}

float computeVariance(float* arr, int n) {
  float mean = 0;
  for (int i = 0; i < n; i++) mean += arr[i];
  mean /= n;
  float var = 0;
  for (int i = 0; i < n; i++) var += (arr[i] - mean) * (arr[i] - mean);
  return var / n;
}
