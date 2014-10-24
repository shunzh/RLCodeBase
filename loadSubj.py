import util
mat = util.loadmat('subj25.parsed.mat')

objDist = mat['pRes'][0].obstDist1
objAngle = mat['pRes'][0].obstAngle1
targDist = mat['pRes'][0].targDist1
targAngle = mat['pRes'][0].targAngle1
action = mat['pRes'][0].action

print objDist, targDist, action
