import inverseModularRL
import modularAgents
import continuousWorld, humanWorld

def main():
  w = inverseModularRL.humanWorldExperiment(range(24, 25))

  qLearnOpts = {'gamma': 0.9,
                'alpha': 0.5,
                'epsilon': 0}
  qFuncs = modularAgents.getHumanWorldContinuousFuncs()
  aHat = modularAgents.ModularAgent(**qLearnOpts)
  aHat.setQFuncs(qFuncs)
  aHat.setWeights(w) # get the weights in the result

  # plot domain and policy
  init = continuousWorld.loadFromMat('miniRes25.mat', 24)
  mdp = humanWorld.HumanWorld(init)

  win = continuousWorld.drawDomain(mdp)

  win.getMouse()
  win.close()

if __name__ == '__main__':
  main()
