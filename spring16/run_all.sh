for simulation in line plane parabola paraboloid sine
do
    nohup ./gp_gpy.py 20 $simulation > /dev/null 2>&1 &
done
