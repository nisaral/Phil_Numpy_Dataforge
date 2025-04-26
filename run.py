import os
import sys
from app import app

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 