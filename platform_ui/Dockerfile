# Version 1.2 - With corrected nginx proxy order
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy setup script
COPY env.sh /docker-entrypoint.d/40-env.sh
RUN chmod +x /docker-entrypoint.d/40-env.sh

# Copy frontend files to Nginx public directory
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
