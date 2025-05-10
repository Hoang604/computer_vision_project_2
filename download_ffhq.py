# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Download Flickr-Face-HQ (FFHQ) dataset to a specified base directory."""

import os
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # avoid "Decompressed Data Too Large" error

#----------------------------------------------------------------------------
# Define the base directory for all downloads and outputs
BASE_DOWNLOAD_DIR = '/media/tuannl1/heavy_weight/data/cv_data'
#----------------------------------------------------------------------------

json_spec_orig = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842, file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')

tfrecords_specs_orig = [
    dict(file_url='https://drive.google.com/uc?id=1LnhoytWihRRJ7CfhLQ76F8YxwxRDlZN3', file_path='tfrecords/ffhq/ffhq-r02.tfrecords', file_size=6860000,      file_md5='63e062160f1ef9079d4f51206a95ba39'),
    dict(file_url='https://drive.google.com/uc?id=1LWeKZGZ_x2rNlTenqsaTk8s7Cpadzjbh', file_path='tfrecords/ffhq/ffhq-r03.tfrecords', file_size=17290000,     file_md5='54fb32a11ebaf1b86807cc0446dd4ec5'),
    dict(file_url='https://drive.google.com/uc?id=1Lr7Tiufr1Za85HQ18yg3XnJXstiI2BAC', file_path='tfrecords/ffhq/ffhq-r04.tfrecords', file_size=57610000,     file_md5='7164cc5531f6828bf9c578bdc3320e49'),
    dict(file_url='https://drive.google.com/uc?id=1LnyiayZ-XJFtatxGFgYePcs9bdxuIJO_', file_path='tfrecords/ffhq/ffhq-r05.tfrecords', file_size=218890000,    file_md5='050cc7e5fd07a1508eaa2558dafbd9ed'),
    dict(file_url='https://drive.google.com/uc?id=1Lt6UP201zHnpH8zLNcKyCIkbC-aMb5V_', file_path='tfrecords/ffhq/ffhq-r06.tfrecords', file_size=864010000,    file_md5='90bedc9cc07007cd66615b2b1255aab8'),
    dict(file_url='https://drive.google.com/uc?id=1LwOP25fJ4xN56YpNCKJZM-3mSMauTxeb', file_path='tfrecords/ffhq/ffhq-r07.tfrecords', file_size=3444980000,   file_md5='bff839e0dda771732495541b1aff7047'),
    dict(file_url='https://drive.google.com/uc?id=1LxxgVBHWgyN8jzf8bQssgVOrTLE8Gv2v', file_path='tfrecords/ffhq/ffhq-r08.tfrecords', file_size=13766900000,  file_md5='74de4f07dc7bfb07c0ad4471fdac5e67'),
    dict(file_url='https://drive.google.com/uc?id=1M-ulhD5h-J7sqSy5Y1njUY_80LPcrv3V', file_path='tfrecords/ffhq/ffhq-r09.tfrecords', file_size=55054580000,  file_md5='05355aa457a4bd72709f74a81841b46d'),
    dict(file_url='https://drive.google.com/uc?id=1M11BIdIpFCiapUqV658biPlaXsTRvYfM', file_path='tfrecords/ffhq/ffhq-r10.tfrecords', file_size=220205650000, file_md5='bf43cab9609ab2a27892fb6c2415c11b'),
]

license_specs_orig = {
    'json':      dict(file_url='https://drive.google.com/uc?id=1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX', file_path='LICENSE.txt',                    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'images':    dict(file_url='https://drive.google.com/uc?id=1sP2qz8TzLkzG2gjwAa4chtdB31THska4', file_path='images1024x1024/LICENSE.txt',    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'thumbs':    dict(file_url='https://drive.google.com/uc?id=1iaL1S381LS10VVtqu-b2WfF9TiY75Kmj', file_path='thumbnails128x128/LICENSE.txt',  file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'wilds':     dict(file_url='https://drive.google.com/uc?id=1rsfFOEQvkd6_Z547qhpq5LhDl2McJEzw', file_path='in-the-wild-images/LICENSE.txt', file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'tfrecords': dict(file_url='https://drive.google.com/uc?id=1SYUmqKdLoTYq-kqsnPsniLScMhspvl5v', file_path='tfrecords/ffhq/LICENSE.txt',     file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
}

# --- Modify paths to include BASE_DOWNLOAD_DIR ---
json_spec = json_spec_orig.copy()
json_spec['file_path'] = os.path.join(BASE_DOWNLOAD_DIR, json_spec_orig['file_path'])

tfrecords_specs = []
for spec_orig in tfrecords_specs_orig:
    spec = spec_orig.copy()
    spec['file_path'] = os.path.join(BASE_DOWNLOAD_DIR, spec_orig['file_path'])
    tfrecords_specs.append(spec)

license_specs = {}
for key, spec_orig in license_specs_orig.items():
    spec = spec_orig.copy()
    spec['file_path'] = os.path.join(BASE_DOWNLOAD_DIR, spec_orig['file_path'])
    license_specs[key] = spec
# --- End of path modifications for global specs ---

#----------------------------------------------------------------------------
def download_file(session, file_spec, stats, chunk_size=128, num_attempts=10):
    # file_path is already an absolute path including BASE_DOWNLOAD_DIR
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    # Get the directory part of the absolute file_path
    file_dir = os.path.dirname(file_path)
    # Create a temporary file path in the same directory
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    
    # Create target directory if it doesn't exist
    # This will create subdirectories under BASE_DOWNLOAD_DIR as needed
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            print(f"\nDownloading {file_path} (attempt {num_attempts - attempts_left}/{num_attempts})...") # Added print for individual file
            with session.get(file_url, stream=True) as res:
                res.raise_for_status() # Raise an exception for bad status codes
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10): # chunk_size is in KB, so multiply by 1024
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)
                        with stats['lock']:
                            stats['bytes_done'] += len(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            if 'pixel_size' in file_spec or 'pixel_md5' in file_spec: # For image files
                with PIL.Image.open(tmp_path) as image:
                    if 'pixel_size' in file_spec and list(image.size) != file_spec['pixel_size']:
                        raise IOError('Incorrect pixel size', file_path)
                    if 'pixel_md5' in file_spec and hashlib.md5(np.array(image)).hexdigest() != file_spec['pixel_md5']:
                        raise IOError('Incorrect pixel MD5', file_path)
            break # Download and validation successful

        except Exception as e: # Catch any exception during download or validation
            with stats['lock']:
                stats['bytes_done'] -= data_size # Rollback byte count if download failed or was partial

            print(f"\nError downloading {file_path}: {e}") # Print error
            
            # Last attempt => raise error.
            if not attempts_left:
                print(f"Failed to download {file_path} after {num_attempts} attempts.")
                raise # Re-raise the last exception

            # Handle Google Drive virus checker nag.
            # This specific handling might occur if the file is small and Google Drive shows an intermediate page.
            if data_size > 0 and data_size < 8192:
                try:
                    with open(tmp_path, 'rb') as f_tmp_read: # Open in read-binary mode
                        data = f_tmp_read.read()
                    # Try to find a new download link in the content (specific to Google Drive's nag screen)
                    links = [html.unescape(link) for link in data.decode('utf-8', errors='ignore').split('"') if 'export=download' in link]
                    if len(links) == 1:
                        file_url = requests.compat.urljoin(file_url, links[0]) # Update URL and retry
                        print(f"Google Drive nag detected, retrying with new URL: {file_url}")
                        continue # Retry with the new URL
                except Exception as parse_e:
                    print(f"Error trying to handle Google Drive nag screen for {file_path}: {parse_e}")
            
            print(f"Retrying download for {file_path} ({attempts_left} attempts left)...")
            time.sleep(1) # Wait a bit before retrying


    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # Atomic operation if possible
    with stats['lock']:
        stats['files_done'] += 1
    print(f"\nSuccessfully downloaded and validated {file_path}")

    # Attempt to clean up any leftover temps (e.g., if script was interrupted before os.replace)
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass # Ignore errors during cleanup

#----------------------------------------------------------------------------

def choose_bytes_unit(num_bytes):
    """Chooses an appropriate unit (B, kB, MB, GB, TB) for displaying byte counts."""
    b = int(np.rint(num_bytes))
    if b < (100 << 0): return 'B', (1 << 0)
    if b < (100 << 10): return 'kB', (1 << 10)
    if b < (100 << 20): return 'MB', (1 << 20)
    if b < (100 << 30): return 'GB', (1 << 30)
    return 'TB', (1 << 40)

#----------------------------------------------------------------------------

def format_time(seconds):
    """Formats a duration in seconds into a human-readable string (e.g., 1h 02m)."""
    s = int(np.rint(seconds))
    if s < 60: return '%ds' % s
    if s < 60 * 60: return '%dm %02ds' % (s // 60, s % 60)
    if s < 24 * 60 * 60: return '%dh %02dm' % (s // (60 * 60), (s // 60) % 60)
    if s < 100 * 24 * 60 * 60: return '%dd %02dh' % (s // (24 * 60 * 60), (s // (60 * 60)) % 24)
    return '>100d' # For very long durations

#----------------------------------------------------------------------------

def download_files(file_specs, num_threads=32, status_delay=0.2, timing_window=50, **download_kwargs):
    """Downloads a list of files in parallel using multiple threads."""

    # Determine which files to download.
    # file_path in spec is already absolute
    done_specs = {spec['file_path']: spec for spec in file_specs if os.path.isfile(spec['file_path'])}
    missing_specs = [spec for spec in file_specs if spec['file_path'] not in done_specs]
    
    files_total = len(file_specs)
    # Calculate total bytes only from file_specs that have 'file_size' (licenses might not, though they do here)
    bytes_total = sum(spec.get('file_size', 0) for spec in file_specs)
    
    stats = dict(
        files_done=len(done_specs), 
        bytes_done=sum(spec.get('file_size', 0) for spec in done_specs.values()), 
        lock=threading.Lock()
    )

    if not missing_specs: # If all files are already downloaded
        print('All %d files already downloaded -- skipping.' % files_total)
        return

    print(f"Need to download {len(missing_specs)} files out of {files_total} total.")

    # Launch worker threads.
    spec_queue = queue.Queue()
    exception_queue = queue.Queue()
    for spec in missing_specs:
        spec_queue.put(spec)
    
    thread_kwargs = dict(spec_queue=spec_queue, exception_queue=exception_queue, stats=stats, download_kwargs=download_kwargs)
    active_threads = []
    for _thread_idx in range(min(num_threads, len(missing_specs))):
        thread = threading.Thread(target=_download_thread, kwargs=thread_kwargs, daemon=True)
        thread.start()
        active_threads.append(thread)

    # Monitor status until done.
    bytes_unit, bytes_div = choose_bytes_unit(bytes_total if bytes_total > 0 else 1) # Avoid division by zero if bytes_total is 0
    spinner = '/-\\|' # Simple text spinner
    timing = [] # For calculating bandwidth and ETA

    while True:
        with stats['lock']:
            files_done = stats['files_done']
            bytes_done = stats['bytes_done']
        
        # Update spinner
        spinner = spinner[1:] + spinner[:1]
        
        # Update timing window for bandwidth calculation
        current_time = time.perf_counter()
        timing = timing[max(len(timing) - timing_window + 1, 0):] + [(current_time, bytes_done)]
        
        bandwidth = 0
        if len(timing) > 1 and (timing[-1][0] - timing[0][0]) > 1e-8 : # Check for valid time diff
             bandwidth = max((timing[-1][1] - timing[0][1]) / (timing[-1][0] - timing[0][0]), 0)
        
        bandwidth_unit, bandwidth_div = choose_bytes_unit(bandwidth)
        
        eta_str = '...'
        if bytes_total > 0 and bytes_done < bytes_total and bandwidth > 1e-8 and len(timing) >= timing_window:
            eta_seconds = (bytes_total - bytes_done) / bandwidth
            eta_str = format_time(eta_seconds)
        elif bytes_total == bytes_done or files_done == files_total : # If download complete
            eta_str = 'done'
        
        # Progress bar display
        progress_percent = (bytes_done / bytes_total * 100) if bytes_total > 0 else 100.0 if files_done == files_total else 0.0

        print('\r%s %6.2f%% done  %d/%d files  %-13s  %-10s  ETA: %-7s ' % (
            spinner[0],
            progress_percent,
            files_done, files_total,
            '%.2f/%.2f %s' % (bytes_done / bytes_div, bytes_total / bytes_div, bytes_unit),
            '%.2f %s/s' % (bandwidth / bandwidth_div, bandwidth_unit),
            eta_str,
        ), end='', flush=True)

        # Check if all files are done (based on file count, as bytes_total might be an estimate if some sizes are missing)
        if files_done == files_total:
            print() # Newline after progress bar finishes
            break

        # Check for exceptions from worker threads
        try:
            exc_info = exception_queue.get(timeout=status_delay) # Wait for status_delay seconds
            # An exception occurred in a thread, re-raise it in the main thread
            print("\nAn error occurred in a download thread:")
            raise exc_info[1].with_traceback(exc_info[2])
        except queue.Empty:
            pass # No exception, continue monitoring
        except KeyboardInterrupt:
            print("\nDownload interrupted by user.")
            # Optionally, signal threads to stop, though daemon threads will exit when main thread exits.
            break 
            
    # Ensure all threads have completed (though daemon threads might not need explicit join if main ends)
    for thread in active_threads:
        thread.join(timeout=5.0) # Wait for threads to finish, with a timeout


def _download_thread(spec_queue, exception_queue, stats, download_kwargs):
    """Worker thread function to download files from the queue."""
    with requests.Session() as session: # Create a session per thread
        while not spec_queue.empty():
            try:
                spec = spec_queue.get_nowait() # Get a file spec without blocking
            except queue.Empty:
                break # Queue is empty, thread can exit
            
            try:
                download_file(session, spec, stats, **download_kwargs)
            except Exception: # Catch any exception from download_file
                exception_queue.put(sys.exc_info()) # Put exception info into the queue for the main thread
                break # Stop this thread on error to prevent cascading failures or to allow main thread to handle
            
            # spec_queue.task_done() # If using queue.join() later, but not strictly necessary with current loop logic

#----------------------------------------------------------------------------

def print_statistics(json_data):
    """Prints statistics about the dataset categories, licenses, and countries."""
    print("\nCalculating dataset statistics...")
    categories = defaultdict(int)
    licenses = defaultdict(int)
    countries = defaultdict(int)
    
    num_total_items = len(json_data)
    if num_total_items == 0:
        print("No data in JSON to calculate statistics.")
        return

    for item in json_data.values(): # json_data here is the processed one with absolute paths
        categories[item['category']] += 1
        licenses[item['metadata']['license']] += 1
        country = item['metadata']['country']
        countries[country if country else '<Unknown>'] += 1

    # Group small country categories into '<Other>'
    for name in [name for name, num in list(countries.items()) if num / num_total_items < 1e-3 and name != '<Unknown>']:
        countries['<Other>'] += countries.pop(name)

    # Prepare rows for table-like printing
    rows = [[]] * 2 # Add some empty rows for spacing
    rows += [['Category', 'Images', '% of all']]
    rows += [['---'] * 3] # Separator
    for name, num in sorted(categories.items(), key=lambda x: -x[1]): # Sort by count descending
        rows += [[name, '%d' % num, '%.2f%%' % (100.0 * num / num_total_items)]]

    rows += [[]] * 2
    rows += [['License', 'Images', '% of all']]
    rows += [['---'] * 3]
    for name, num in sorted(licenses.items(), key=lambda x: -x[1]):
        rows += [[name, '%d' % num, '%.2f%%' % (100.0 * num / num_total_items)]]

    rows += [[]] * 2
    rows += [['Country', 'Images', '% of all', '% of known']]
    rows += [['---'] * 4]
    num_known_country_items = num_total_items - countries.get('<Unknown>', 0)
    for name, num in sorted(countries.items(), key=lambda x: (-x[1] if x[0] not in ['<Unknown>', '<Other>'] else x[0] == '<Unknown>')): # Complex sort for <Unknown> and <Other> last
        percent_of_all = 100.0 * num / num_total_items
        percent_of_known = 0
        if name != '<Unknown>' and num_known_country_items > 0:
            percent_of_known = 100.0 * num / num_known_country_items
        rows += [[name, '%d' % num, '%.2f%%' % percent_of_all, '%.2f%%' % percent_of_known if name != '<Unknown>' else 'N/A']]
        
    rows += [[]] * 2
    
    # Calculate column widths for neat printing
    widths = [max(len(str(cell)) for cell in column if cell is not None) for column in itertools.zip_longest(*rows)]
    for row in rows:
        print("  ".join(str(cell).ljust(width) if cell is not None else " " * width for cell, width in zip(row, widths)))
    print()

#----------------------------------------------------------------------------

def recreate_aligned_images(json_data, dst_dir_suffix='realign1024x1024', output_size=1024, transform_size=4096, enable_padding=True):
    """Recreates aligned 1024x1024 images from in-the-wild images and landmarks."""
    
    # Construct the full destination directory path using BASE_DOWNLOAD_DIR
    dst_dir = os.path.join(BASE_DOWNLOAD_DIR, dst_dir_suffix)
    print(f'Recreating aligned images in {dst_dir}...')
    
    if dst_dir: # Ensure dst_dir is not empty or None
        os.makedirs(dst_dir, exist_ok=True)
        # Copy the main license file to the new aligned images directory
        # license_specs['json']['file_path'] is already an absolute path
        main_license_path = license_specs['json']['file_path']
        if os.path.isfile(main_license_path):
            shutil.copyfile(main_license_path, os.path.join(dst_dir, 'LICENSE.txt'))
            print(f"Copied LICENSE.txt to {dst_dir}")
        else:
            print(f"Warning: Main license file {main_license_path} not found. Cannot copy to {dst_dir}.")

    num_items = len(json_data)
    for item_idx, item in enumerate(json_data.values()): # item already has absolute paths
        print('\rProcessing image %d / %d ... ' % (item_idx + 1, num_items), end='', flush=True)

        # Parse landmarks.
        # pylint: disable=unused-variable
        try:
            lm = np.array(item['in_the_wild']['face_landmarks'])
        except KeyError:
            print(f"\nSkipping item_idx {item_idx}: 'face_landmarks' not found in 'in_the_wild' data.")
            continue
            
        # ... (Landmark definitions: lm_chin, lm_eyebrow_left, etc. remain the same)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise


        # Calculate auxiliary vectors for alignment.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x_norm = np.hypot(*x)
        if x_norm == 0: # Avoid division by zero if x is a zero vector
             print(f"\nSkipping item_idx {item_idx}: Cannot normalize vector x for crop rectangle (norm is zero). Landmarks might be problematic.")
             continue
        x /= x_norm
        
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        # src_file is already an absolute path from the processed json_data
        src_file = item['in_the_wild']['file_path']
        if not os.path.isfile(src_file):
            print(f'\nCannot find source image: {src_file}. Please ensure "--wilds" task was run successfully.')
            # Original script returns here, which stops all further alignment.
            # Depending on desired behavior, one might 'continue' to the next item instead.
            # For now, keeping original behavior:
            return 
        try:
            img = PIL.Image.open(src_file)
            if img.mode != 'RGB': # Convert to RGB if not already
                img = img.convert('RGB')
        except PIL.UnidentifiedImageError:
            print(f"\nSkipping item_idx {item_idx}: Cannot open or identify image file {src_file}.")
            continue
        except Exception as e:
            print(f"\nSkipping item_idx {item_idx}: Error opening image {src_file}: {e}")
            continue


        # Shrink image if qsize is much larger than output_size, for efficiency.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            if rsize[0] > 0 and rsize[1] > 0: # Ensure dimensions are positive
                img = img.resize(rsize, PIL.Image.Resampling.LANCZOS) # Changed ANTIALIAS to LANCZOS (recommended for Pillow 9+)
                quad /= shrink
                qsize /= shrink
            else:
                print(f"\nSkipping item_idx {item_idx}: Invalid resize dimensions after shrink for {src_file}.")
                continue


        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            if crop[2] > crop[0] and crop[3] > crop[1]: # Ensure crop dimensions are valid
                img = img.crop(crop)
                quad -= crop[0:2]
            else:
                print(f"\nSkipping item_idx {item_idx}: Invalid crop dimensions for {src_file}.")
                continue


        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img_array = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img_array.shape
            y_grid, x_grid, _ = np.ogrid[:h, :w, :1] # Renamed y, x to y_grid, x_grid to avoid conflict
            
            # Calculate mask ensuring denominators are not zero
            mask_x_denom1 = pad[0] if pad[0] > 0 else 1
            mask_x_denom2 = pad[2] if pad[2] > 0 else 1
            mask_y_denom1 = pad[1] if pad[1] > 0 else 1
            mask_y_denom2 = pad[3] if pad[3] > 0 else 1

            mask = np.maximum(1.0 - np.minimum(np.float32(x_grid) / mask_x_denom1, np.float32(w-1-x_grid) / mask_x_denom2), 
                              1.0 - np.minimum(np.float32(y_grid) / mask_y_denom1, np.float32(h-1-y_grid) / mask_y_denom2))
            
            blur = qsize * 0.02
            # Apply Gaussian filter only if blur > 0
            if blur > 1e-5: # Check for a small positive blur value
                 img_array += (scipy.ndimage.gaussian_filter(img_array, [blur, blur, 0]) - img_array) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img_array += (np.median(img_array, axis=(0,1)) - img_array) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img_array), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        if img.size[0] > 0 and img.size[1] > 0: # Ensure image has positive dimensions before transform
            img = img.transform((transform_size, transform_size), PIL.Image.Transform.QUAD, (quad + 0.5).flatten(), PIL.Image.Resampling.BILINEAR) # Changed PIL.Image.QUAD to PIL.Image.Transform.QUAD
            if output_size < transform_size:
                img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS) # Changed ANTIALIAS
        else:
            print(f"\nSkipping item_idx {item_idx}: Image has zero dimension before transform for {src_file}.")
            continue


        # Save aligned image.
        # dst_subdir is constructed based on the absolute dst_dir
        # Original logic: item_idx - item_idx % 1000 gives 0 for 0-999, 1000 for 1000-1999, etc.
        # This creates directories like /00000/, /01000/, etc.
        dst_subdir_name = '%05d' % ((item_idx // 1000) * 1000) 
        dst_subdir = os.path.join(dst_dir, dst_subdir_name)
        os.makedirs(dst_subdir, exist_ok=True)
        
        output_filename = '%05d.png' % item_idx
        output_filepath = os.path.join(dst_subdir, output_filename)
        try:
            img.save(output_filepath)
        except Exception as e:
            print(f"\nError saving aligned image {output_filepath} for item_idx {item_idx}: {e}")


    # All done.
    print('\rProcessing image %d / %d ... done' % (num_items, num_items))

#----------------------------------------------------------------------------

def run(tasks, **download_kwargs):
    """Main function to orchestrate downloading and processing tasks."""
    
    # Ensure the BASE_DOWNLOAD_DIR itself exists
    print(f"Ensuring base download directory exists: {BASE_DOWNLOAD_DIR}")
    os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

    # Check for JSON metadata and its license file (paths are already absolute and prefixed)
    if not os.path.isfile(json_spec['file_path']) or not os.path.isfile(license_specs['json']['file_path']):
        print(f"Downloading JSON metadata to {json_spec['file_path']} and/or license to {license_specs['json']['file_path']}...")
        download_files([json_spec, license_specs['json']], **download_kwargs)
    else:
        print(f"JSON metadata ({json_spec['file_path']}) and its license ({license_specs['json']['file_path']}) already exist.")


    print(f"Parsing JSON metadata from {json_spec['file_path']}...")
    try:
        with open(json_spec['file_path'], 'rb') as f:
            json_data_raw = json.load(f, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        print(f"ERROR: JSON metadata file not found at {json_spec['file_path']}. Please ensure it's downloaded correctly.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {json_spec['file_path']}. The file might be corrupted.")
        return


    # IMPORTANT: Prefix file_path within the loaded json_data to make them absolute
    print(f"Prefixing file paths in loaded JSON data with base directory: {BASE_DOWNLOAD_DIR}...")
    json_data_processed = OrderedDict()
    for key, item_value_orig in json_data_raw.items():
        # Create a deep copy if there are nested mutable structures that will be modified,
        # but for just updating file_path strings, a shallow copy of the item dict is usually enough.
        # The original script modifies dicts in place, which is fine if item_value_orig is not reused elsewhere.
        # Here, we make a copy of the item's dictionary to be safe.
        item_value = item_value_orig.copy() 

        if 'image' in item_value and isinstance(item_value['image'], dict) and 'file_path' in item_value['image']:
            item_value['image'] = item_value['image'].copy() # Copy the 'image' dict before modifying
            item_value['image']['file_path'] = os.path.join(BASE_DOWNLOAD_DIR, item_value_orig['image']['file_path'])
        
        if 'thumbnail' in item_value and isinstance(item_value['thumbnail'], dict) and 'file_path' in item_value['thumbnail']:
            item_value['thumbnail'] = item_value['thumbnail'].copy() # Copy the 'thumbnail' dict
            item_value['thumbnail']['file_path'] = os.path.join(BASE_DOWNLOAD_DIR, item_value_orig['thumbnail']['file_path'])
            
        if 'in_the_wild' in item_value and isinstance(item_value['in_the_wild'], dict) and 'file_path' in item_value['in_the_wild']:
            item_value['in_the_wild'] = item_value['in_the_wild'].copy() # Copy the 'in_the_wild' dict
            item_value['in_the_wild']['file_path'] = os.path.join(BASE_DOWNLOAD_DIR, item_value_orig['in_the_wild']['file_path'])
            
        json_data_processed[key] = item_value
    
    # Use the processed json_data (with absolute paths) from now on
    json_data = json_data_processed


    if 'stats' in tasks:
        print_statistics(json_data) # This function doesn't use file paths directly for I/O

    specs_to_download = [] # List of file specifications for download_files function
    # Note: file_paths in license_specs and tfrecords_specs are already absolute
    # file_paths in json_data (item['image'], etc.) are now also absolute

    if 'images' in tasks:
        start_image_index = 27989
        all_image_specs = [item['image'] for item in list(json_data.values()) if 'image' in item and item['image']]
        
        if start_image_index < len(all_image_specs):
            images_to_download_this_run = all_image_specs[start_image_index:]
            specs_to_download += images_to_download_this_run
            print(f"Đã thêm {len(images_to_download_this_run)} ảnh vào hàng đợi tải xuống, bắt đầu từ ảnh thứ {start_image_index + 1} (chỉ số {start_image_index}).")
        else:
            print(f"Chỉ số bắt đầu ({start_image_index}) vượt quá tổng số ảnh ({len(all_image_specs)}). Sẽ không có ảnh nào được tải xuống cho tác vụ này.")
        
        # Luôn thêm tệp giấy phép cho hình ảnh nếu tác vụ hình ảnh được chọn
        specs_to_download += [license_specs['images']]
# ...existing code...
    if 'thumbs' in tasks:
        specs_to_download += [item['thumbnail'] for item in json_data.values() if 'thumbnail' in item and item['thumbnail']] # Ensure item['thumbnail'] exists
        specs_to_download += [license_specs['thumbs']]
    if 'wilds' in tasks:
        specs_to_download += [item['in_the_wild'] for item in json_data.values() if 'in_the_wild' in item and item['in_the_wild']] # Ensure item['in_the_wild'] exists
        specs_to_download += [license_specs['wilds']]
    if 'tfrecords' in tasks:
        specs_to_download += tfrecords_specs # These are already full specs with absolute paths
        specs_to_download += [license_specs['tfrecords']]

    if specs_to_download: 
        print(f"Preparing to download files for selected tasks. Target base directory: {BASE_DOWNLOAD_DIR}")
        np.random.shuffle(specs_to_download) 
        download_files(specs_to_download, **download_kwargs)
    elif tasks and any(t in ['images', 'thumbs', 'wilds', 'tfrecords'] for t in tasks):
        # This condition might be met if 'images' was a task but start_image_index was too high
        if not ('images' in tasks and start_image_index >= len([item['image'] for item in list(json_data.values()) if 'image' in item and item['image']])):
             print("No files to download for the specified tasks (images, thumbs, wilds, tfrecords). They might already exist or JSON data is empty.")


    if 'align' in tasks:
        # recreate_aligned_images will use BASE_DOWNLOAD_DIR internally
        # and json_data which now has absolute paths for source 'in_the_wild' images
        recreate_aligned_images(json_data, **download_kwargs) # Pass relevant download_kwargs if needed by align, though it doesn't use them directly

#----------------------------------------------------------------------------

def run_cmdline(argv):
    """Parses command line arguments and runs the specified tasks."""
    parser = argparse.ArgumentParser(prog=argv[0], description=f'Download Flickr-Face-HQ (FFHQ) dataset. Files will be stored under {BASE_DOWNLOAD_DIR}.')
    
    # Task arguments
    parser.add_argument('-j', '--json',         help='download metadata as JSON (254 MB)', dest='tasks', action='append_const', const='json') # Though json is always downloaded if missing
    parser.add_argument('-s', '--stats',        help='print statistics about the dataset', dest='tasks', action='append_const', const='stats')
    parser.add_argument('-i', '--images',       help='download 1024x1024 images as PNG (89.1 GB)', dest='tasks', action='append_const', const='images')
    parser.add_argument('-t', '--thumbs',       help='download 128x128 thumbnails as PNG (1.95 GB)', dest='tasks', action='append_const', const='thumbs')
    parser.add_argument('-w', '--wilds',        help='download in-the-wild images as PNG (955 GB)', dest='tasks', action='append_const', const='wilds')
    parser.add_argument('-r', '--tfrecords',    help='download multi-resolution TFRecords (273 GB)', dest='tasks', action='append_const', const='tfrecords')
    parser.add_argument('-a', '--align',        help='recreate 1024x1024 images from in-the-wild images', dest='tasks', action='append_const', const='align')
    
    # Download configuration arguments
    parser.add_argument('--num_threads',        help='number of concurrent download threads (default: 32)', type=int, default=32, metavar='NUM')
    parser.add_argument('--status_delay',       help='time between download status prints (default: 0.2 sec)', type=float, default=0.2, metavar='SEC')
    parser.add_argument('--timing_window',      help='samples for estimating download eta (default: 50)', type=int, default=50, metavar='LEN')
    parser.add_argument('--chunk_size',         help='chunk size for each download thread in KB (default: 128 KB)', type=int, default=128, metavar='KB')
    parser.add_argument('--num_attempts',       help='number of download attempts per file (default: 10)', type=int, default=10, metavar='NUM')

    args = parser.parse_args(argv[1:])
    
    if not args.tasks:
        # If only --json is implicitly handled, ensure tasks list is not empty for run()
        # The run() function handles initial JSON download if not present.
        # For clarity, if no tasks are specified, we can print help or a message.
        # However, the original script exits if args.tasks is empty.
        # Let's make 'json' an implicit task if nothing else is specified,
        # or simply rely on run() to fetch it.
        # For now, matching original behavior:
        print('No tasks specified. Please see "-h" for help.')
        sys.exit(1) # Changed exit(1) to sys.exit(1)

    # Prepare kwargs for the run function, separating task list from download params
    run_kwargs = vars(args).copy()
    tasks_to_run = run_kwargs.pop('tasks', []) # Get tasks and remove from kwargs
    
    # The 'json' task is a bit special as it's a prerequisite.
    # The run function handles downloading json_spec and license_specs['json'] if they are missing,
    # regardless of whether 'json' is explicitly in tasks.
    # If 'json' is the *only* task, it will just download these two files.
    
    print(f"Selected tasks: {tasks_to_run}")
    print(f"Download parameters: Threads={args.num_threads}, ChunkSizeKB={args.chunk_size}, Attempts={args.num_attempts}")

    run(tasks=tasks_to_run, **run_kwargs) # Pass tasks and other args (like num_threads)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_cmdline(sys.argv)

#----------------------------------------------------------------------------
