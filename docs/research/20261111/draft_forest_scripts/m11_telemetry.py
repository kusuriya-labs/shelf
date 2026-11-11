#!/usr/bin/env python3
"""Sample read-only Linux sysfs clocks and GPU device users during M11."""
import glob,json,os,subprocess,time
from pathlib import Path
out=Path(os.environ.get('M11_OUTPUT','/tmp/forest-m11'))
out.mkdir(parents=True,exist_ok=True)
dev=Path('/sys/class/drm/card2/device')
hw=next(iter((dev/'hwmon').glob('hwmon*')))
def read(p):
 try: return p.read_text().strip()
 except OSError: return None
with (out/'telemetry.jsonl').open('a') as f:
 while not Path('/tmp/forest-m11.exit').exists():
  p=subprocess.run(['fuser','/dev/kfd','/dev/dri/renderD130'],capture_output=True,text=True)
  pids=sorted(set(p.stdout.split()))
  processes={pid:read(Path('/proc')/pid/'cmdline') for pid in pids}
  d={'processes':processes,'time':time.time(),'busy':read(dev/'gpu_busy_percent'),'vram':read(dev/'mem_info_vram_used'),'sclk':read(dev/'pp_dpm_sclk'),'mclk':read(dev/'pp_dpm_mclk'),'temp_mC':read(hw/'temp1_input'),'power_uW':read(hw/'power1_average'),'device_pids':p.stdout.split()}
  f.write(json.dumps(d)+'\n'); f.flush(); time.sleep(2)
