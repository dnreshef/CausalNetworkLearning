for simulation in line plane parabola paraboloid sine corrugated_curve
do
    nohup ./gp_gpy.py 150 $simulation > /dev/null 2>&1 &
done
