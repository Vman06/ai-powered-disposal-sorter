from gpiozero import LED
from time import sleep

# BCM numbers for physical pins 11, 13, 15
LED_TRASH   = LED(17)  # red  (pin 11)
LED_RECYCLE = LED(27)  # yellow (pin 13)
LED_COMPOST = LED(22)  # green (pin 15)

def all_off():
    LED_TRASH.off()
    LED_RECYCLE.off()
    LED_COMPOST.off()

try:
    while True:
        # blink all together
        LED_TRASH.on(); LED_RECYCLE.on(); LED_COMPOST.on()
        sleep(0.5)
        all_off()
        sleep(0.5)

        # chase pattern (optional)
        LED_TRASH.on(); sleep(0.25); LED_TRASH.off()
        LED_RECYCLE.on(); sleep(0.25); LED_RECYCLE.off()
        LED_COMPOST.on(); sleep(0.25); LED_COMPOST.off()

except KeyboardInterrupt:
    all_off()
    print("\nStopped.")
