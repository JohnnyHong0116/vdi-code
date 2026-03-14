#include <Adafruit_NeoPixel.h>

#define LED_PIN 6
#define SWITCH_PIN 2
#define NUM_LEDS 11

const uint8_t GLOBAL_BRIGHTNESS = 100; // modify for led brightness

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setAll(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();
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

  if (attached) {
    setAll(0, 255, 0);   // green
  } else {
    setAll(0, 0, 255);   // blue
  }

  delay(100);
}