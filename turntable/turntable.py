import time
import serial


class Turntable:
    """
    Python controller for Arduino + DM542 + stepper motor turntable.

    Arduino firmware should support commands:
        HOME
        POS
        SPEED <rpm>
        ROT <degree>
        GOTO <degree>

    Communication protocol:
        Python sends one command ending with '\n'
        Arduino executes the command
        Arduino replies with:
            READY
            DONE ...
            POS ...
            ERR ...
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 20.0,
        reset_wait: float = 2.0,
    ):
        """
        Open serial connection to Arduino.

        Args:
            port: Arduino serial port, e.g. "/dev/cu.usbmodem1101"
            baudrate: Must match Arduino Serial.begin(115200)
            timeout: Max time to wait for Arduino response
            reset_wait: Arduino usually resets when serial port opens
        """
        self.port = port
        self.baudrate = baudrate

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=timeout,
        )

        # Opening serial usually resets Arduino.
        # Wait for Arduino to reboot.
        time.sleep(reset_wait)

        # Clear old buffered messages if any.
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_command(self, command: str) -> str:
        """
        Send one command to Arduino and wait for one response.

        Args:
            command: Command string, e.g. "ROT 90"

        Returns:
            Arduino response string.

        Raises:
            TimeoutError: if Arduino does not reply.
            RuntimeError: if Arduino returns ERR.
        """
        command = command.strip()

        # Send command with newline.
        # Arduino code reads until '\n'.
        self.ser.write((command + "\n").encode("utf-8"))
        self.ser.flush()

        # Wait for one response line.
        response = self.ser.readline().decode("utf-8", errors="ignore").strip()

        if not response:
            raise TimeoutError(f"No response from Arduino after command: {command}")

        if response.startswith("ERR"):
            raise RuntimeError(f"Arduino error after command '{command}': {response}")

        return response

    def home(self) -> str:
        """
        Define current physical position as 0 degree.

        Note:
            This does not physically search for a home sensor.
            You must manually place the turntable at the desired 0° position first.
        """
        return self.send_command("HOME")

    def get_position(self) -> float:
        """
        Query current software angle from Arduino.

        Returns:
            Current angle in degrees, normalized to [0, 360).
        """
        response = self.send_command("POS")

        # Expected response: "POS 90.000"
        parts = response.split()
        if len(parts) != 2 or parts[0] != "POS":
            raise RuntimeError(f"Unexpected POS response: {response}")

        return float(parts[1])

    def set_speed(self, rpm: float) -> str:
        """
        Set turntable rotation speed.

        Args:
            rpm: revolutions per minute.

        Example:
            15 RPM means:
                one full turn = 4 seconds
                90 degrees = about 1 second
        """
        return self.send_command(f"SPEED {rpm}")

    def rotate(self, degrees: float) -> str:
        """
        Relative rotation.

        Args:
            degrees:
                +90 means rotate 90 degrees in positive direction.
                -90 means rotate 90 degrees in opposite direction.
        """
        return self.send_command(f"ROT {degrees}")

    def goto(self, degrees: float) -> str:
        """
        Absolute angle control.

        Args:
            degrees:
                Target absolute angle relative to HOME.
                Example: 0, 90, 180, 270.

        Arduino will move to this target angle and return DONE after motion finishes.
        """
        return self.send_command(f"GOTO {degrees}")

    def __enter__(self):
        """Allows: with Turntable(...) as tt:"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically close serial connection."""
        self.close()