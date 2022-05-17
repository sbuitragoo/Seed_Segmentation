import errno 
import os 
from os import path 
import time as time
import argparse
from gpiozero import LED # funcion led Del GPIO 
import cv2


class CaptureSession():

    def __init__(self,path,step,duration):
        self.path = path
        self.step = self.code_time(step)
        self.duration = self.code_time(duration)
        self.flash = LED(17) # pin Para conectar Flash
        self.final=self.duration/self.step # Cantidad de capturas

    def code_time(self, time): 
        """Casts a minute/hour/day time to seconds

        Args:
            time (str): time in a format different to seconds

        Returns:
            float: time casted to seconds
        """

        if time[-1] == 'd': timeSeg = float(time[:-1])*86400 
        elif time[-1] == 'h': timeSeg = float(time[:-1])*3600
        elif time[-1] == 'm': timeSeg = float(time[:-1])*60
        else: timeSeg = float(time) 

        return timeSeg

    def makePath(self):
        """Creates the path where the captures will be saved
        """
        try: 
            os.mkdir(path) 
        except OSError as e: 
            if e.errno == errno.EEXIST: 
                pass 
    
    def capture(self):
        """Makes the capture session during the time specified in the constructor.
        """
        cont=0
        while(cont<=self.final):
            self.flash.off()  # ensender Flash
            time.sleep(1)
            fileName = self.path+"/S"+time.strftime("%d-%m-%Y,%H:%M:%S")+".jpg"
            os.system(f"raspistill -o {fileName}")
            self.flash.on() # Apagar Flash
            time.sleep(0.2)
            cont=cont+1 
            time.sleep(self.step-2) # Esperar step -2 seguntos

    def StartCapture(self):
        """Initializes the capture session methods
        """
        self.makePath()
        self.capture()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    manual = subparser.add_parser('manual')
    auto = subparser.add_parser('auto')

    auto.add_argument('--path', type=str, required=True, 
                        help="Path where the session is going to be saved")
    auto.add_argument('--step', type=str, required=True,
                        help="Step between captures")
    auto.add_argument('--duration', type=str, required=True,
                        help="Entire capture session duration")

    args = parser.parse_args()
    
    if args.command == 'manual':
        cs = CaptureSession(path="capturas", step="30m", duration="10d")
        cs.StartCapture()


    elif args.command == 'auto':
        cs = CaptureSession(path=args.path, step=args.step, duration=args.duration)
        cs.StartCapture()