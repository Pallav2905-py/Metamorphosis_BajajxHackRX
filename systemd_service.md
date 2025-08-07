# Production Deployment with Systemd Service

## Create Systemd Service File

```bash
sudo nano /etc/systemd/system/metamorphosis-api.service
```

Add the following content:

```ini
[Unit]
Description=BajajxHackRX Metamorphosis API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Metamorphosis_BajajxHackRX
Environment=PATH=/home/ubuntu/Metamorphosis_BajajxHackRX/.venv/bin
ExecStart=/home/ubuntu/Metamorphosis_BajajxHackRX/.venv/bin/python app.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=metamorphosis-api

[Install]
WantedBy=multi-user.target
```

## Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable metamorphosis-api

# Start the service
sudo systemctl start metamorphosis-api

# Check status
sudo systemctl status metamorphosis-api

# View logs
sudo journalctl -u metamorphosis-api -f
```

## Service Management Commands

```bash
# Start service
sudo systemctl start metamorphosis-api

# Stop service
sudo systemctl stop metamorphosis-api

# Restart service
sudo systemctl restart metamorphosis-api

# Check status
sudo systemctl status metamorphosis-api

# View logs
sudo journalctl -u metamorphosis-api --since "1 hour ago"
```
