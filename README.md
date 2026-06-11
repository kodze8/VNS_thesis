## VNS Explorer Setup Instructions

---

### 1. Clone the repository
```bash
git clone https://github.com/kodze8/VNS_thesis.git
```

### 2. Move into the project directory
```bash
cd VNS_thesis
```

### 3. Create a virtual environment
```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (cmd)**
```cmd
venv\Scripts\activate.bat
```

### 5. Upgrade pip
```bash
pip install --upgrade pip
```

### 6. Install dependencies
```bash
pip install -r requirements.txt
```

### 7. Navigate to source directory
```bash
cd src
```

### 8. Run the application
```bash
PORT=5000 python app.py
```

---

### Notes
- Default server runs on port 5000  
- If port 5000 does not work or is already in use, you can run with a different port like:

```bash
PORT=5001 python app.py
```

or any other available port:

```bash
PORT=6000 python app.py
```
