"""Draw a circle with the galvo laser.  New standalone script - does not modify test.py.

Uses exactly the same serial protocol as test.py (the SC point list ending in "M11eR", "S" to stop):
    SC
    X<x>Y<y>F<speed>W<wait>E      one line per point
    ...
    M11eR                         laser on + run the list

Run:     python draw_circle.py                 (defaults below, Enter to stop)
         python draw_circle.py --radius 8000 --points 48
         python draw_circle.py --dry-run       (print the command, send nothing, laser stays off)

The mirror grid is 0..65535 with 32768 in the middle.  Because the projected rectangle is warped,
the circle will look warped the same way (pincushion / keystone of the optics), that is expected.
"""
import argparse
import math

# ----------------------------------------------------------------------------- settings
PORT = "COM5"           # the CP210x USB-UART bridge (COM3 is a Bluetooth port)
BAUD = 1000000
CENTER_X = 32768
CENTER_Y = 32768
RADIUS = 10000          # mirror units (full grid half-width is 32768)
POINTS = 36             # polygon corners; more = rounder circle
SPEED = 4000            # F value, same as test.py
WAIT = 3000             # W value, same as test.py's rectangle; 0 = leave W out of every point
# -----------------------------------------------------------------------------


def circle_points(cx, cy, radius, n):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x = int(round(cx + radius * math.cos(a)))
        y = int(round(cy + radius * math.sin(a)))
        pts.append((min(max(x, 0), 65535), min(max(y, 0), 65535)))   # stay inside 0..65535
    return pts


def circle_message(cx=CENTER_X, cy=CENTER_Y, radius=RADIUS, n=POINTS, speed=SPEED, wait=WAIT):
    w = f"W{wait}" if wait else ""
    lines = [f"X{x}Y{y}F{speed}{w}E" for x, y in circle_points(cx, cy, radius, n)]
    return "SC\n" + "\n".join(lines) + "\nM11eR"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--radius", type=int, default=RADIUS, help="mirror units")
    ap.add_argument("--points", type=int, default=POINTS)
    ap.add_argument("--speed", type=int, default=SPEED)
    ap.add_argument("--wait", type=int, default=WAIT)
    ap.add_argument("--cx", type=int, default=CENTER_X)
    ap.add_argument("--cy", type=int, default=CENTER_Y)
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit; nothing is sent")
    a = ap.parse_args()

    if a.points < 3:
        raise SystemExit("--points must be >= 3")
    if a.radius <= 0 or a.radius > 32768:
        raise SystemExit("--radius must be 1..32768")

    msg = circle_message(a.cx, a.cy, a.radius, a.points, a.speed, a.wait)
    if a.dry_run:
        print(msg)
        print(f"\n[dry run] {a.points} points, radius {a.radius}, centre ({a.cx}, {a.cy}); nothing sent")
        return

    import serial   # imported here so --dry-run works without the device / pyserial

    ser = serial.Serial(a.port, baudrate=BAUD)
    try:
        ser.write(bytes(msg, "utf-8"))          # starts the laser and the circle
        print(f"Drawing a circle on {a.port} (radius {a.radius}, {a.points} points).")
        input("Press Enter to stop... ")
    finally:
        ser.write(bytes("S", "utf-8"))          # same stop command as test.py
        ser.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
