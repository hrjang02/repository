#!/bin/csh

cd /data03/EMIS_NEA9km/EMIS_MONTHLY/UNIMIXv3.1_PNR

sshpass -p "보낼곳pw" scp -P 17773 -r *.202112_w_3days_SpinUp.ncf 사용자id@보낼서버ip:/data04/보낼곳디렉토리

exit(0)