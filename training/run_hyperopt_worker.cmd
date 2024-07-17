REM Append to the python path our custom function directories
REM E:\PyCharm\COM6906-Dissertation\model
REM E:\PyCharm\COM6906-Dissertation\util

set PYTHONPATH=$PYTHONPATH;E:\PyCharm\COM6906-Dissertation\model;E:\PyCharm\COM6906-Dissertation\util

hyperopt-mongo-worker --mongo=localhost:27017/hyperopt --poll-interval=0.1 --reserve-timeout=1800