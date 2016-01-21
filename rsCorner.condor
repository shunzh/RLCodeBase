Universe = vanilla
Executable = /usr/local/bin/bash
+Project = "AI_ROBOTICS"
+Group = "grad"
Notification = Never
Notify_user = menie482@cs.utexas.edu
Output     = rsCorner_out.$(Process)
Error      = rsCorner_err.$(Process)

num_of_processes=200

Arguments = rs.sh $(Process) '-t corner'
Queue $(num_of_processes)
