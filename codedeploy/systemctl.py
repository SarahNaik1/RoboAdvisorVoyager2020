/etc/systemd/system

[Unit]
Description=A service to start/stop robo-advisor-service
After=network.target

[Service]
WorkingDirectory=/opt/voyager/app
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target