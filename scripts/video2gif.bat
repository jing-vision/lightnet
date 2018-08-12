REM video2gif /path/to/video/file

set palette=palette.png
set filters=fps=15,scale=320:-1:flags=lanczos

ffmpeg -v warning -i %1 -vf %filters%,palettegen -y %palette%
ffmpeg -v warning -i %1 -i %palette% -lavfi %filters%" [x]; [x][1:v] paletteuse" -y %1.gif