# pip install --pre --extra-index-url https://archive.panda3d.org/ panda3d
# start pythonw -B world.py

from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import Sequence
from direct.actor.Actor import Actor
from direct.stdpy import threading

class AgentThread(threading.Thread):
    def __init__(self, world):
        threading.Thread.__init__(self)
        self.world = world

    def run(self):
        import time
        for i in range(100):
            for j in range(20):
                self.world.act('pos', 'left')
                time.sleep(0.05)
            for j in range(20):
                self.world.act('pos', 'right')
                time.sleep(0.05)
            for j in range(20):
                self.world.act('pos', 'forward')
                time.sleep(0.05)
            for j in range(20):
                self.world.act('pos', 'backward')
                time.sleep(0.05)
            for j in range(360):
                self.world.act('hpr', 'left')
                time.sleep(0.02)
            for j in range(360):
                self.world.act('hpr', 'right')
                time.sleep(0.02)

class World(object):
    def __init__(self):
        loadPrcFileData('', 'window-type none')
        base = ShowBase()
        base.makeDefaultPipe(printPipeTypes=False)
        base.openDefaultWindow()
 
        windowProperties = WindowProperties()
        windowProperties.setTitle('TPJ')
        windowProperties.setSize(1600,800)
        base.win.requestProperties(windowProperties)
        base.setBackgroundColor(0, 0, 0)

        render = NodePath('render')

        ground = render.attachNewNode(CardMaker('ground').generate())
        ground.setTexture(loader.loadTexture('maps/envir-ground.jpg'))
        ground.setScale(400)
        ground.setPos(-200,-200,-1)
        ground.setHpr(0, 270, 0)

        box = loader.loadModel('box')
        box.reparentTo(render)   
        box.setTexture(loader.loadTexture('maps/envir-cylinder.png'), 1)
        box.setScale(8)
        box.setPos(-90, 0, 0)

        ball = loader.loadModel('smiley.egg')
        ball.reparentTo(render)
        ball.setTexture(loader.loadTexture('maps/noise.rgb'), 1)
        ball.setScale(6)
        ball.setPos(+90, 0, 6)

        teapot = loader.loadModel('teapot')  #froweny
        teapot.reparentTo(render)
        teapot.setTexture(loader.loadTexture('maps/envir-treetrunk.jpg'))
        teapot.setScale(2.7)
        teapot.setPos(0, +90, 0)

        panda = Actor("models/panda-model", {"walk": "models/panda-walk4"})
        panda.reparentTo(render)
        panda.setScale(0.025)
        panda.setPos(0, -90, 0)
        panda.setHpr(180, 0, 0)
        panda.loop("walk")

        agent = Actor("models/panda", {"walk": "models/panda-walk"})
        agent.reparentTo(render)
        agent.setScale(2,2, 2)
        agent.setPos(0, 0, 1)
        agent.setHpr(0, 30, 0)
        agent.loop("walk")

        cameraBirdView = render.attachNewNode(Camera('camera-birdview'))
        cameraBirdView.setPos(0, -150, 200)
        cameraBirdView.setHpr(0, -50, 0)
        displayRegionBirdView = base.win.makeDisplayRegion(1.0, 1.0, 1.0, 1.0)
        displayRegionBirdView.setCamera(cameraBirdView)
        lensBirdView = OrthographicLens()
        lensBirdView.setFilmSize(+450, +450)
        lensBirdView.setNearFar(-430, +430) 
        cameraBirdView.node().setLens(lensBirdView)

        cameraFirstPerson = render.attachNewNode(Camera('camera-firstpersonview'))
        cameraFirstPerson.setPos(0, 0, 0)
        cameraFirstPerson.setHpr(-180, 0, 0)
        displayRegionFirstPerson = base.win.makeDisplayRegion(1.0, 1.0, 1.0, 1.0)
        displayRegionFirstPerson.setCamera(cameraFirstPerson)        
        cameraFirstPerson.reparentTo(agent)

        displayRegionBirdView = cameraBirdView.node().getDisplayRegion(0)
        displayRegionFirstPerson = cameraFirstPerson.node().getDisplayRegion(0)
        displayRegionBirdView.setDimensions(0, 0.6, 0, 1)
        displayRegionFirstPerson.setDimensions(0.6, 1, 0, 1)
        cameraBirdView.node().getLens().setAspectRatio(float(displayRegionBirdView.getPixelWidth()) / float(displayRegionBirdView.getPixelHeight()))
        cameraFirstPerson.node().getLens().setAspectRatio(float(displayRegionFirstPerson.getPixelWidth()) / float(displayRegionFirstPerson.getPixelHeight()))

        self.mysee = displayRegionFirstPerson
        self.myposX = 0
        self.myposY = 0
        self.myposZ = 0
        self.myhprX = 0
        self.myhprY = 0
        self.myhprZ = 0  
        self.myrun = agent

        AgentThread(self).start()

        base.run()

    def see(self):
        base.movie(namePrefix='./tpj_see', duration=2, fps=1, sd=8, format='png', source=self.mysee)

    def act(self, action, parameter):
        if action=='pos':
            if parameter=='left':
                self.myposX = self.myposX -1
            elif parameter=='right':
                self.myposX = self.myposX +1
            elif parameter=='forward':
                self.myposY = self.myposY -1
            elif parameter=='backward':
                self.myposY = self.myposY +1
            else:
                pass
        elif action=='hpr':
            if parameter=='left':
                self.myhprX = self.myhprX -1
            elif parameter=='right':
                self.myhprX = self.myhprX +1
            else:
                pass
        else:
            pass
        self.myrun.setPos(self.myposX, self.myposY, self.myposZ)
        self.myrun.setHpr(self.myhprX, self.myhprY, self.myhprZ)

if __name__ == '__main__':
    world = World()
