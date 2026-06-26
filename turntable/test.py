from turntable import Turntable
import time


PORT = "/dev/cu.usbmodem1101"


def main():
    with Turntable(PORT) as tt:
        print("Set current position as HOME")
        print(tt.home())

        print("Set speed")
        print(tt.set_speed(15))

        print("Current position:")
        print(tt.get_position())

        print("Go to 90 deg")
        print(tt.goto(90))
        print("Position:", tt.get_position())

        time.sleep(1)

        print("Go to 180 deg")
        print(tt.goto(180))
        print("Position:", tt.get_position())
        time.sleep(1)

        print("Go to 270 deg")
        print(tt.goto(270))
        print("Position:", tt.get_position())
        time.sleep(1)

        print("Back to HOME / 0 deg")
        print(tt.goto(0))
        print("Position:", tt.get_position())



if __name__ == "__main__":
    main()