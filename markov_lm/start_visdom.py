set -e
nohup visdom -env_path test_data -cache_type JPWA -port 6006 &> visdom.log &
echo PID started: $( jobs -p ) 
