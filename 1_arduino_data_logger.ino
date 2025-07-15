#define SIGNAL_PIN A0

const int SAMPLE_RATE = 10;
const int BUFFER_SIZE = 10;
const unsigned long SAMPLE_INTERVAL = 1000 / SAMPLE_RATE;

unsigned long lastSampleTime = 0;
int sampleBuffer[BUFFER_SIZE];
int bufferIndex = 0;
int rawValue = 0;
float voltage = 0.0;
float smoothedValue = 0.0;
float baseline = 0.0;
bool calibrated = false;
unsigned long calibrationStart = 0;
const unsigned long CALIBRATION_TIME = 10000;

float minValue = 1023.0;
float maxValue = 0.0;
float totalSamples = 0;
float runningSum = 0;

void setup() {
  Serial.begin(9600);
  
  for(int i = 0; i < BUFFER_SIZE; i++) {
    sampleBuffer[i] = 0;
  }
  
  Serial.println("# Radio Telescope Data Logger");
  Serial.println("# Starting calibration...");
  Serial.println("# Timestamp(ms),Raw_ADC,Voltage(V),Smoothed,Baseline_Diff,Signal_Strength");
  
  calibrationStart = millis();
}

void loop() {
  unsigned long currentTime = millis();
  
  if(currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;
    
    rawValue = analogRead(SIGNAL_PIN);
    voltage = (rawValue * 5.0) / 1023.0;
    
    sampleBuffer[bufferIndex] = rawValue;
    bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
    
    int sum = 0;
    for(int i = 0; i < BUFFER_SIZE; i++) {
      sum += sampleBuffer[i];
    }
    smoothedValue = sum / (float)BUFFER_SIZE;
    
    if (!calibrated) {
      if (currentTime - calibrationStart < CALIBRATION_TIME) {
        runningSum += smoothedValue;
        totalSamples++;
      } else {
        baseline = runningSum / totalSamples;
        calibrated = true;
        Serial.print("# Calibration complete. Baseline: ");
        Serial.println(baseline);
        minValue = smoothedValue;
        maxValue = smoothedValue;
      }
    }
    
    if(calibrated) {
      if(smoothedValue < minValue) minValue = smoothedValue;
      if(smoothedValue > maxValue) maxValue = smoothedValue;
      
      float baselineDiff = smoothedValue - baseline;
      float signalStrength = abs(baselineDiff);
      
      Serial.print(currentTime);
      Serial.print(",");
      Serial.print(rawValue);
      Serial.print(",");
      Serial.print(voltage, 3);
      Serial.print(",");
      Serial.print(smoothedValue, 2);
      Serial.print(",");
      Serial.print(baselineDiff, 2);
      Serial.print(",");
      Serial.println(signalStrength, 2);
      
      static int sampleCount = 0;
      sampleCount++;
      if(sampleCount >= 100) {
        Serial.print("# Stats - Min: ");
        Serial.print(minValue, 2);
        Serial.print(", Max: ");
        Serial.print(maxValue, 2);
        Serial.print(", Range: ");
        Serial.println(maxValue - minValue, 2);
        sampleCount = 0;
      }
    }
  }
  
  delay(1);
}