import pulp

def lp(S, A, R, psi, trans):
  """
  Args:
    S: state set
    A: action set
    R: reward candidate set
    psi: prior belief on rewards
    trans: transition function
  """

