#include <Adafruit_NeoPixel.h>

#define LED_PIN 6
#define SWITCH_PIN 2
#define NUM_LEDS 11

const uint8_t GLOBAL_BRIGHTNESS = 100; // modify for led brightness
const unsigned long REPORT_PERIOD_MS = 50;
const unsigned long STROBE_PERIOD_MS = 200;
const unsigned long BLUE_PULSE_PERIOD_MS = 120;
const uint8_t BLUE_PULSE_LEVELS[] = {40, 80, 130, 180, 230, 255, 230, 180, 130, 80};
const uint8_t BLUE_PULSE_COUNT = sizeof(BLUE_PULSE_LEVELS) / sizeof(BLUE_PULSE_LEVELS[0]);
const unsigned long RED_PULSE_PERIOD_MS = 120;
const uint8_t RED_PULSE_LEVELS[] = {30, 60, 100, 150, 205, 255, 205, 150, 100, 60};
const uint8_t RED_PULSE_COUNT = sizeof(RED_PULSE_LEVELS) / sizeof(RED_PULSE_LEVELS[0]);

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);
char currentCommand = '\0';
unsigned long lastReportMs = 0;
unsigned long lastStrobeToggleMs = 0;
unsigned long lastBluePulseMs = 0;
unsigned long lastRedPulseMs = 0;
bool strobeOn = false;
uint8_t bluePulseIndex = 0;
uint8_t redPulseIndex = 0;

void setAll(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();
}

void applyCommand(char cmd, bool attached) {
  switch (cmd) {
    case 'r':
      setAll(255, 0, 0);
      break;
    case 'g':
      setAll(0, 255, 0);
      break;
    case 'b':
      setAll(0, 0, 255);
      break;
    case 'y':
      setAll(255, 180, 0);
      break;
    case 'e':
      setAll(0, 0, 0);
      break;
    case 's':
      if (millis() - lastStrobeToggleMs >= STROBE_PERIOD_MS) {
        lastStrobeToggleMs = millis();
        strobeOn = !strobeOn;
      }
      if (strobeOn) {
        setAll(0, 0, 255);
      } else {
        setAll(0, 0, 0);
      }
      break;
    case 'p':
      if (millis() - lastBluePulseMs >= BLUE_PULSE_PERIOD_MS) {
        lastBluePulseMs = millis();
        bluePulseIndex = (bluePulseIndex + 1) % BLUE_PULSE_COUNT;
      }
      setAll(0, 0, BLUE_PULSE_LEVELS[bluePulseIndex]);
      break;
    case 'a':
      if (millis() - lastRedPulseMs >= RED_PULSE_PERIOD_MS) {
        lastRedPulseMs = millis();
        redPulseIndex = (redPulseIndex + 1) % RED_PULSE_COUNT;
      }
      setAll(RED_PULSE_LEVELS[redPulseIndex], 0, 0);
      break;
    default:
      if (attached) {
        setAll(0, 255, 0);
      } else {
        setAll(0, 0, 255);
      }
      break;
  }
}

void setup() {
  pinMode(SWITCH_PIN, INPUT_PULLUP);
  Serial.begin(9600);

  strip.begin();
  strip.setBrightness(GLOBAL_BRIGHTNESS);
  strip.show();
}

void loop() {
  bool attached = (digitalRead(SWITCH_PIN) == LOW);
  while (Serial.available() > 0) {
    char incoming = (char)Serial.read();
    if (incoming == '\n' || incoming == '\r') {
      continue;
    }
    currentCommand = incoming;
  }

  applyCommand(currentCommand, attached);

  if (millis() - lastReportMs >= REPORT_PERIOD_MS) {
    lastReportMs = millis();
    Serial.print("0.0,");
    Serial.println(attached ? 1 : 0);
  }

  delay(10);
}
