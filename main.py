from run.server_run import app
from model.paths import INPUT_PATH, OUTPUT_PATH

import os



if __name__ == '__main__':
    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    app.run(debug=True, host='0.0.0.0')