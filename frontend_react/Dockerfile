# Stage 1: Build
FROM node:22-alpine as builder
WORKDIR /app
COPY package*.json ./
# Relaxed install (ignoring peer deps conflicts if any persist)
RUN npm install --legacy-peer-deps
COPY . .
# Skip type checking during build for stability
RUN npm run build

# Stage 2: Serve
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
