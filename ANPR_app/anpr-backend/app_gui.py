import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import subprocess
import requests
import time
import sys
import os
import io
import base64

API_URL = "http://127.0.0.1:8000"

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def kill_existing_api():
    try:
        result = subprocess.run(["lsof", "-ti", ":8000"], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid])
                append_log(f"Killed existing process on port 8000 (PID {pid})")
        time.sleep(1)
    except:
        pass

def start_api():
    api_path = resource_path("api.py")
    venv_python = os.environ.get("VENV_PYTHON", "")
    if not venv_python or not os.path.exists(venv_python):
        venv_python = os.path.join(os.path.dirname(api_path), ".venv311", "bin", "python3")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    process = subprocess.Popen(
        [python_exec, "-u", api_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=os.path.dirname(api_path)
    )
    def stream_logs():
        for line in process.stdout:
            line = line.strip()
            if line:
                append_log(line)
    threading.Thread(target=stream_logs, daemon=True).start()
    return process

def append_log(message):
    def _update():
        log_box.config(state=tk.NORMAL)
        if "✓" in message or "loaded" in message.lower() or "ready" in message.lower():
            tag = "success"
        elif "error" in message.lower() or "failed" in message.lower():
            tag = "error"
        elif "loading" in message.lower() or "downloading" in message.lower():
            tag = "info"
        elif "warning" in message.lower():
            tag = "warning"
        else:
            tag = "normal"
        log_box.insert(tk.END, message + "\n", tag)
        log_box.see(tk.END)
        log_box.config(state=tk.DISABLED)
    root.after(0, _update)

def wait_for_api(timeout=120):
    for _ in range(timeout * 2):
        try:
            r = requests.get(f"{API_URL}/health", timeout=1)
            data = r.json()
            if data.get("detection_model_loaded") and data.get("vgg19_loaded"):
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def pick_and_detect():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "ANPR_dataset")

    if not os.path.exists(dataset_dir):
        dataset_dir = script_dir

    file_path = filedialog.askopenfilename(
        title="Select License Plate Image",
        initialdir=dataset_dir,
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    detect_btn.config(state=tk.DISABLED, text="Processing...")
    append_log(f"→ {os.path.basename(file_path)}")

    def run_detection():
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{API_URL}/detect",
                    files={"file": (os.path.basename(file_path), f, "image/jpeg")},
                    timeout=60
                )
            response.raise_for_status()
            result = response.json()
            num = result.get("num_detections", 0)
            append_log(f"✓ Done — {num} plate(s) found")
            root.after(0, lambda: show_result_window(file_path, result))
        except Exception as e:
            append_log(f"✗ Error: {str(e)}")
            root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            root.after(0, lambda: detect_btn.config(state=tk.NORMAL, text="Select Image"))
    threading.Thread(target=run_detection, daemon=True).start()


def show_result_window(file_path, result):
    all_detections = result.get("detections", [])
    if all_detections:
        detections = [max(all_detections, key=lambda d: d.get("confidence", 0))]
    else:
        detections = []

    win = tk.Toplevel()
    win.title("Result")
    win.configure(bg="white")
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    win.geometry(f"{sw}x{sh}+0+0")
    win.resizable(True, True)

    PAD = 28

    tk.Frame(win, bg="#e0e0e0", height=1).pack(fill=tk.X)
    topbar = tk.Frame(win, bg="white", pady=12)
    topbar.pack(fill=tk.X, padx=PAD)
    tk.Label(topbar, text="Detection Result",
             font=("Helvetica", 16, "bold"), fg="#111111", bg="white").pack(side=tk.LEFT)
    tk.Button(topbar, text="Close", command=win.destroy,
              font=("Helvetica", 11), bg="white", fg="#111111",
              activebackground="#f0f0f0", relief=tk.FLAT,
              highlightbackground="#cccccc", highlightthickness=1,
              padx=20, pady=6, cursor="hand2").pack(side=tk.RIGHT)
    tk.Frame(win, bg="#e0e0e0", height=1).pack(fill=tk.X)

    if not detections:
        tk.Label(win, text="No plates detected.",
                 font=("Helvetica", 13), fg="#ef4444", bg="white").pack(pady=40)
        return

    det          = detections[0]
    plate_text   = det.get("plate_text", "UNREADABLE")
    ocr_conf     = det.get("ocr_confidence", 0)
    det_conf     = det.get("confidence", 0)
    num_chars    = det.get("num_characters", 0)
    is_readable  = det.get("is_readable", False)
    char_details = det.get("character_details", [])
    plate_b64    = det.get("plate_image", "")
    accent       = "#22c55e" if is_readable else "#ef4444"

    main = tk.Frame(win, bg="white")
    main.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=12)

    # LEFT: original image
    left_col = tk.Frame(main, bg="white")
    left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

    tk.Label(left_col, text="INPUT IMAGE",
             font=("Helvetica", 9), fg="#999999", bg="white").pack(anchor="w", pady=(0, 6))

    img = Image.open(file_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        fnt = ImageFont.load_default()

    for d in detections:
        b = d.get("bbox", {})
        x1, y1, x2, y2 = b.get("x1",0), b.get("y1",0), b.get("x2",0), b.get("y2",0)
        pt   = d.get("plate_text", "?")
        conf = d.get("confidence", 0)
        col  = "#22c55e" if d.get("is_readable") else "#ef4444"
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        lbl  = f"{pt}  {conf:.1%}"
        tw   = len(lbl) * 10
        draw.rectangle([x1, y1-26, x1+tw+8, y1], fill=col)
        draw.text((x1+4, y1-23), lbl, fill="white", font=fnt)

    max_img_w = int(sw * 0.60)
    max_img_h = int(sh * 0.84)
    iw, ih = img.size
    scale = min(max_img_w / iw, max_img_h / ih, 1.0)
    img = img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)

    photo = ImageTk.PhotoImage(img)
    img_lbl = tk.Label(left_col, image=photo, bg="white", anchor="nw")
    img_lbl.image = photo
    img_lbl.pack(anchor="nw")

    tk.Frame(main, bg="#e0e0e0", width=1).pack(side=tk.LEFT, fill=tk.Y)

    # RIGHT: results
    right_col = tk.Frame(main, bg="white", width=int(sw * 0.35))
    right_col.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 0))
    right_col.pack_propagate(False)

    tk.Label(right_col, text=plate_text,
             font=("Helvetica", 38, "bold"), fg=accent, bg="white").pack(anchor="w", pady=(0, 4))
    tk.Label(right_col, text=f"Detection  {det_conf:.1%}",
             font=("Helvetica", 11), fg="#666666", bg="white").pack(anchor="w")
    tk.Label(right_col, text=f"OCR  {ocr_conf:.1%}",
             font=("Helvetica", 11), fg="#666666", bg="white").pack(anchor="w")
    tk.Label(right_col, text=f"Characters  {num_chars}",
             font=("Helvetica", 11), fg="#666666", bg="white").pack(anchor="w")

    tk.Frame(right_col, bg="#e0e0e0", height=1).pack(fill=tk.X, pady=14)

    tk.Label(right_col, text="PLATE CROP",
             font=("Helvetica", 9), fg="#999999", bg="white").pack(anchor="w", pady=(0, 6))

    if plate_b64:
        try:
            b64d = plate_b64.split(",")[1] if "," in plate_b64 else plate_b64
            pimg = Image.open(io.BytesIO(base64.b64decode(b64d))).convert("RGB")
            MAX_W = int(sw * 0.30)
            pw, ph = pimg.size
            if pw > MAX_W:
                pimg = pimg.resize((MAX_W, int(ph * MAX_W / pw)), Image.LANCZOS)
            pp = ImageTk.PhotoImage(pimg)
            pl = tk.Label(right_col, image=pp, bg="white", anchor="w")
            pl.image = pp
            pl.pack(anchor="w")
        except:
            tk.Label(right_col, text="(unavailable)", fg="#cccccc", bg="white",
                     font=("Helvetica", 9)).pack(anchor="w")

    tk.Frame(right_col, bg="#e0e0e0", height=1).pack(fill=tk.X, pady=14)

    tk.Label(right_col, text="CHARACTER BREAKDOWN",
             font=("Helvetica", 9), fg="#999999", bg="white").pack(anchor="w", pady=(0, 8))

    chars_outer = tk.Frame(right_col, bg="white")
    chars_outer.pack(anchor="w", fill=tk.X)

    chars_canvas = tk.Canvas(chars_outer, bg="white", highlightthickness=0, height=160)
    chars_hbar = tk.Scrollbar(chars_outer, orient="horizontal", command=chars_canvas.xview)
    chars_canvas.configure(xscrollcommand=chars_hbar.set)
    chars_hbar.pack(side=tk.BOTTOM, fill=tk.X)
    chars_canvas.pack(side=tk.TOP, fill=tk.X)

    chars_row = tk.Frame(chars_canvas, bg="white")
    chars_canvas.create_window((0, 0), window=chars_row, anchor="nw")
    chars_row.bind("<Configure>",
                   lambda e: chars_canvas.configure(scrollregion=chars_canvas.bbox("all")))

    if char_details:
        for cd in char_details:
            char     = cd.get("character", "?")
            conf     = cd.get("confidence", 0)
            c_b64    = cd.get("image", "")
            conf_pct = conf * 100
            c_color  = "#22c55e" if conf >= 0.90 else ("#f59e0b" if conf >= 0.65 else "#ef4444")

            cc = tk.Frame(chars_row, bg="white", padx=6, pady=6,
                          highlightbackground="#e0e0e0", highlightthickness=1)
            cc.pack(side=tk.LEFT, padx=3)

            if c_b64:
                try:
                    b64d = c_b64.split(",")[1] if "," in c_b64 else c_b64
                    cimg = Image.open(io.BytesIO(base64.b64decode(b64d))).convert("RGB")
                    TARGET_H = 72
                    cw, ch = cimg.size
                    new_w = max(1, int(cw * TARGET_H / ch))
                    cimg = cimg.resize((new_w, TARGET_H), Image.LANCZOS)
                    cp = ImageTk.PhotoImage(cimg)
                    cl = tk.Label(cc, image=cp, bg="white")
                    cl.image = cp
                    cl.pack()
                except:
                    tk.Label(cc, text="", bg="white", width=3, height=4).pack()

            tk.Label(cc, text=char,
                     font=("Helvetica", 13, "bold"), fg="#111111", bg="white").pack(pady=(3, 0))
            tk.Label(cc, text=f"{conf_pct:.1f}%",
                     font=("Helvetica", 9), fg=c_color, bg="white").pack()

            bar_c = tk.Canvas(cc, bg="#e8e8e8", width=44, height=3,
                              highlightthickness=0, bd=0)
            bar_c.pack(pady=(2, 0))
            bar_c.create_rectangle(0, 0, max(2, int(44 * conf)), 3, fill=c_color, outline="")
    else:
        tk.Label(chars_row, text="No character data.",
                 font=("Helvetica", 10), fg="#cccccc", bg="white").pack(anchor="w")


# ══════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════
root = tk.Tk()
root.title("ANPR System")
root.geometry("560x440")
root.configure(bg="white")
root.resizable(False, False)

log_frame = tk.Frame(root, bg="white")
log_frame.pack(padx=16, pady=(16, 8), fill=tk.BOTH, expand=True)

log_box = scrolledtext.ScrolledText(
    log_frame, height=18, width=58,
    bg="#1a1a1a", fg="#cccccc",
    font=("Courier", 10),
    state=tk.DISABLED,
    relief=tk.FLAT, bd=0,
    highlightbackground="#e0e0e0",
    highlightthickness=1,
    insertbackground="#cccccc"
)
log_box.pack(fill=tk.BOTH, expand=True)
log_box.tag_config("success", foreground="#22c55e")
log_box.tag_config("error",   foreground="#f87171")
log_box.tag_config("info",    foreground="#60a5fa")
log_box.tag_config("warning", foreground="#fbbf24")
log_box.tag_config("normal",  foreground="#cccccc")

tk.Frame(root, bg="#e0e0e0", height=1).pack(fill=tk.X)

bottom = tk.Frame(root, bg="white", pady=14)
bottom.pack(fill=tk.X, padx=16)

detect_btn = tk.Button(bottom,
                       text="Select Image",
                       font=("Helvetica", 12, "bold"),
                       bg="white", fg="#111111",
                       activebackground="#f0f0f0", activeforeground="#111111",
                       relief=tk.FLAT,
                       highlightbackground="#cccccc",
                       highlightthickness=1,
                       padx=20, pady=8,
                       state=tk.DISABLED, cursor="hand2",
                       command=pick_and_detect)
detect_btn.pack(side=tk.LEFT)

status_var = tk.StringVar(value="Starting up...")
tk.Label(bottom, textvariable=status_var,
         font=("Helvetica", 10), fg="#999999", bg="white").pack(side=tk.LEFT, padx=12)

def init_sequence():
    append_log("Checking port 8000...")
    kill_existing_api()
    append_log("Starting API server...")
    start_api()
    append_log("Loading models, please wait...")
    ready = wait_for_api(timeout=120)
    if ready:
        append_log("✓ RT-DETRv2 loaded!")
        append_log("✓ VGG19 loaded!")
        append_log("✓ System ready!")
        root.after(0, lambda: [
            status_var.set("Ready"),
            detect_btn.config(state=tk.NORMAL)
        ])
    else:
        append_log("✗ Failed to start")
        root.after(0, lambda: [
            status_var.set("Failed to start"),
            messagebox.showerror("Error",
                "API did not start in time.\nTry running api.py manually first.")
        ])

threading.Thread(target=init_sequence, daemon=True).start()
root.mainloop()