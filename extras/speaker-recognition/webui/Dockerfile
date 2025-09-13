FROM node:18-alpine

# Install curl for health checks
RUN apk add curl

WORKDIR /app

# Accept build arguments for environment variables
ARG VITE_ALLOWED_HOSTS
ENV VITE_ALLOWED_HOSTS=$VITE_ALLOWED_HOSTS

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 5173

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]