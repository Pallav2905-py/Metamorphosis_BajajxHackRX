# Nginx Reverse Proxy Setup (Optional)

## Install Nginx

```bash
sudo apt install nginx -y
```

## Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/metamorphosis-api
```

Add configuration:

```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## Enable Site

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/metamorphosis-api /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Enable Nginx to start on boot
sudo systemctl enable nginx
```

## Update Security Group

Change inbound rule from port 8000 to port 80 (HTTP).

## Access Your API

Your API will now be available at:
- `http://YOUR_EC2_PUBLIC_IP/`
- API docs: `http://YOUR_EC2_PUBLIC_IP/docs`
