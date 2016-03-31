for simulation in line plane parabola paraboloid sine corrugated_curve
do
    ./gp_gpy.py 150 $simulation > /dev/null &
done
