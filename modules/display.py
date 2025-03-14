import ipywidgets as widgets
from IPython.display import display, Audio, clear_output, HTML
import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
import time
import soundfile as sf

def annotate(audio_db):
    """
    Interactive annotation function for audio clips.
    Filters clips based on annotation status and score range.
    Enables random selection of clips, audio playback, and annotation.
    
    Args:
        audio_db: An instance of Audio_DB class containing the clips to annotate
    """
    # Create state variables
    annotation_active = False
    current_index = None
    filtered_df = None
    
    # Create widgets for filtering
    score_range_slider = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=audio_db.score_min,
        max=audio_db.score_max,
        step=0.01,
        description='Score Range:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='500px')
    )
    
    # Button to apply filters
    filter_button = widgets.Button(
        description='Start Annotating',
        button_style='primary',
        tooltip='Filter clips and start annotation process',
        icon='play'
    )
    
    # Stop annotation button
    stop_button = widgets.Button(
        description='Stop Annotating',
        button_style='danger',
        tooltip='Exit annotation mode',
        icon='stop',
        layout=widgets.Layout(width='150px', height='40px')
    )
    
    # Output widget to display filter results
    filter_output = widgets.Output()
    
    # Output widget for displaying the spectrogram
    spectrogram_output = widgets.Output()
    
    # Output widget for displaying audio player and clip info
    audio_info_output = widgets.Output()
    
    # Output widget for annotation status
    status_output = widgets.Output()
    
    # Create annotation buttons
    not_present_button = widgets.Button(
        description='Not Present (0)',
        button_style='danger',
        tooltip='Target sound is not present in the clip',
        layout=widgets.Layout(width='150px', height='50px')
    )
    
    present_button = widgets.Button(
        description='Present (1)',
        button_style='success',
        tooltip='Target sound is present in the clip',
        layout=widgets.Layout(width='150px', height='50px')
    )
    
    uncertain_button = widgets.Button(
        description='Uncertain (3)',
        button_style='warning',
        tooltip='Uncertain if target sound is present',
        layout=widgets.Layout(width='150px', height='50px')
    )
    
    # Create a HBox to layout the annotation buttons
    annotation_buttons = widgets.HBox([
        not_present_button, 
        present_button, 
        uncertain_button
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row',
        justify_content='flex-end',
        width='100%',
        margin='10px 0px'
    ))
    
    # Function to generate mel spectrogram with buffer and markers
    def create_mel_spectrogram(audio_path, clip_start, clip_end, sr=None):
        # Add buffer of up to 1 second on each side (but don't go below 0 for start)
        buffer_sec = 1.0
        buffered_start = max(0, clip_start - buffer_sec)
        
        # Load audio file with buffer
        # We'll determine the file's total duration to make sure we don't exceed it
        y_check, sr_check = librosa.load(audio_path, sr=sr, offset=0, duration=10)  # Just to get sr
        
        # Fixed: Use path parameter instead of filename
        file_duration = librosa.get_duration(path=audio_path, sr=sr_check)
        buffered_end = min(file_duration, clip_end + buffer_sec)
        
        # Load the audio with buffer
        y, sr = librosa.load(audio_path, sr=sr, offset=buffered_start, duration=buffered_end-buffered_start)
        
        # Create figure
        plt.figure(figsize=(12, 4))
        
        # Generate mel spectrogram with increased fmax to 10kHz
        nyquist = sr // 2
        fmax = min(10000, nyquist)  # Use 10kHz or Nyquist, whichever is smaller
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Display mel spectrogram with time showing actual file seconds
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=fmax, 
                                       x_coords=np.linspace(buffered_start, buffered_end, S.shape[1]))
        
        plt.colorbar(img, format='%+2.0f dB')
        plt.xlabel("Time (seconds)")
        
        # Add vertical lines for clip boundaries at actual file times
        plt.axvline(x=clip_start, color='r', linestyle='-', linewidth=2, alpha=0.7)
        plt.axvline(x=clip_end, color='r', linestyle='-', linewidth=2, alpha=0.7)
        
        # Add text labels at the vertical lines
        plt.text(clip_start, 0, f"{clip_start:.1f}s", color='r', fontweight='bold', 
                 verticalalignment='bottom', horizontalalignment='center')
        plt.text(clip_end, 0, f"{clip_end:.1f}s", color='r', fontweight='bold', 
                 verticalalignment='bottom', horizontalalignment='center')
        
        # Set the title to include the clip range information
        plt.title(f'Mel Spectrogram - Clip ({clip_start:.1f}s - {clip_end:.1f}s) - With Buffer')
        plt.tight_layout()
        
        # Return the audio data for the clip without buffer for playback
        clip_start_sample = int((clip_start - buffered_start) * sr)
        clip_end_sample = int((clip_end - buffered_start) * sr)
        
        # Make sure we don't exceed array bounds
        clip_start_sample = max(0, min(clip_start_sample, len(y) - 1))
        clip_end_sample = max(clip_start_sample + 1, min(clip_end_sample, len(y)))
        
        clip_audio = y[clip_start_sample:clip_end_sample]
        
        return clip_audio, sr, y, buffered_start, buffered_end
    
    # Function to refresh filtered data
    def refresh_filtered_data():
        nonlocal filtered_df
        min_score, max_score = score_range_slider.value
        
        # Re-filter to get only clips that still need annotation
        filtered_df = audio_db.df.filter(
            (pl.col("annotation") == 4) & 
            (pl.col("score") >= min_score) & 
            (pl.col("score") <= max_score)
        )
        
        return filtered_df
    
    # Function to get a random clip from filtered data
    def get_random_clip(filtered_data):
        if len(filtered_data) == 0:
            return None, None
        
        # Select a random index from the filtered data
        random_idx = random.randint(0, len(filtered_data) - 1)
        return filtered_data.row(random_idx), random_idx
    
    # Function to display audio and spectrogram
    def display_clip(clip_data):
        # Extract clip details
        file_name = clip_data[0]  # file_name
        file_path = clip_data[1]  # file_path
        clip_start = clip_data[3]  # clip_start
        clip_end = clip_data[4]   # clip_end
        sampling_rate = clip_data[5]  # sampling_rate
        score = clip_data[6]  # score
        
        # Display spectrogram in its output area
        with spectrogram_output:
            clear_output(wait=True)
            
            try:
                # Create spectrogram with buffer and markers, and get audio data
                clip_audio, sr, buffered_audio, buffered_start, buffered_end = create_mel_spectrogram(
                    file_path, clip_start, clip_end, sr=sampling_rate
                )
                plt.show()
            except Exception as e:
                print(f"Error generating spectrogram: {e}")
                print(f"File path: {file_path}")
        
        # Display audio player and clip info in their output area
        with audio_info_output:
            clear_output(wait=True)
            
            try:
                # Create audio widget for playback with autoplay
                display(HTML("<h4>Audio Playback:</h4>"))
                display(Audio(data=clip_audio, rate=sr, autoplay=True))
                
                # Display clip details
                clip_info = f"""
                <div style="margin: 10px 0;">
                    <h4>Clip Information:</h4>
                    <p><b>File:</b> {os.path.basename(file_path)}</p>
                    <p><b>Time Range:</b> {clip_start:.1f}s - {clip_end:.1f}s</p>
                    <p><b>Duration:</b> {clip_end - clip_start:.1f}s</p>
                    <p><b>Sampling Rate:</b> {sampling_rate} Hz</p>
                    <p><b>Score:</b> {score:.3f}</p>
                </div>
                """
                display(HTML(clip_info))
                
            except Exception as e:
                print(f"Error with audio playback: {e}")
                
                # Try alternative approach
                try:
                    print("\nAttempting alternative loading method...")
                    full_audio, sr = librosa.load(file_path, sr=sampling_rate)
                    start_idx = int(clip_start * sr)
                    end_idx = int(clip_end * sr)
                    
                    # Ensure indices are within bounds
                    start_idx = max(0, min(start_idx, len(full_audio) - 1))
                    end_idx = max(start_idx + 1, min(end_idx, len(full_audio)))
                    
                    clip_audio = full_audio[start_idx:end_idx]
                    
                    # Create audio widget for playback
                    display(Audio(data=clip_audio, rate=sr, autoplay=True))
                    print("Audio loaded using alternative method.")
                except Exception as e2:
                    print(f"Alternative method also failed: {e2}")
                    print("Please check the file path and format.")
    
    # Function to update annotation in the dataframe
    def update_annotation(annotation_value):
        nonlocal current_index, filtered_df, annotation_active
        
        if current_index is None or filtered_df is None:
            return
        
        # Get the current clip's file_name for identification
        current_clip = filtered_df.row(current_index)
        file_name = current_clip[0]  # file_name
        file_path = current_clip[1]  # file_path
        clip_start = current_clip[3]  # clip_start
        clip_end = current_clip[4]  # clip_end
        
        # Update the annotation in the Audio_DB dataframe
        # We need to find the matching row and update its annotation
        mask = (audio_db.df["file_name"] == file_name) & \
               (audio_db.df["file_path"] == file_path) & \
               (audio_db.df["clip_start"] == clip_start) & \
               (audio_db.df["clip_end"] == clip_end)
        
        # Create a new column that will have the updated annotation value where the mask is True
        update_col = pl.when(mask).then(annotation_value).otherwise(pl.col("annotation"))
        
        # Update the dataframe
        audio_db.df = audio_db.df.with_columns(update_col.alias("annotation"))
        
        # Important: Re-filter the data to exclude the just-annotated clip
        refresh_filtered_data()
        
        with status_output:
            clear_output(wait=True)
            print(f"✓ Clip annotated as: {annotation_value_to_text(annotation_value)}")
            print(f"Remaining clips to annotate: {len(filtered_df)}")
        
        # If we still have clips to annotate, load the next random clip
        if len(filtered_df) > 0 and annotation_active:
            load_random_clip()
        else:
            annotation_active = False
            with status_output:
                clear_output(wait=True)
                print("✓ All clips have been annotated in the selected range!")
                print("Click 'Start Annotating' to select a new range or continue with additional clips.")
            
            # Clear the displays when done
            with spectrogram_output:
                clear_output(wait=True)
            with audio_info_output:
                clear_output(wait=True)
    
    # Helper function to convert annotation value to text
    def annotation_value_to_text(value):
        if value == 0:
            return "Not Present"
        elif value == 1:
            return "Present"
        elif value == 3:
            return "Uncertain"
        else:
            return f"Unknown ({value})"
    
    # Function to load a random clip
    def load_random_clip():
        nonlocal current_index, filtered_df
        
        # Make sure we're working with the most up-to-date filtered data
        refresh_filtered_data()
        
        if filtered_df is None or len(filtered_df) == 0:
            with status_output:
                clear_output(wait=True)
                print("No clips match the filter criteria or all clips have been annotated.")
            return
        
        # Get a random clip
        clip_data, clip_idx = get_random_clip(filtered_df)
        if clip_data is None:
            with status_output:
                clear_output(wait=True)
                print("No clips remain to be annotated in the selected range.")
            return
            
        current_index = clip_idx
        
        # Display the clip
        display_clip(clip_data)
        
        with status_output:
            clear_output(wait=True)
            print(f"Please annotate this clip ({len(filtered_df)} clips remaining)")
            print("Listen to the audio and examine the spectrogram, then select an annotation.")
    
    # Function to handle filtering and start the annotation process
    def start_annotation(b):
        nonlocal annotation_active, filtered_df
        
        with filter_output:
            clear_output(wait=True)
            
            # Apply filters and get updated filtered data
            filtered_df = refresh_filtered_data()
            
            # Display filtered results stats
            total_clips = len(audio_db.df)
            filtered_clips = len(filtered_df)
            min_score, max_score = score_range_slider.value
            
            print(f"Total clips: {total_clips}")
            print(f"Filtered clips: {filtered_clips} (not reviewed, scores between {min_score:.2f} and {max_score:.2f})")
            
            if filtered_clips > 0:
                annotation_active = True
                print("\nStarting annotation process. Random clips will be selected for review.")
                time.sleep(1)  # Brief pause for feedback
                load_random_clip()
            else:
                print("\nNo clips match the filter criteria. Please adjust the score range.")
                annotation_active = False
    
    # Function to handle stopping annotation
    def stop_annotation(b):
        nonlocal annotation_active
        annotation_active = False
        
        with filter_output:
            clear_output()
            print("Annotation session ended.")
            
            # Get summary of annotation progress
            total = len(audio_db.df)
            not_reviewed = len(audio_db.df.filter(pl.col("annotation") == 4))
            reviewed = total - not_reviewed
            
            print(f"Summary:")
            print(f"Total clips: {total}")
            print(f"Reviewed: {reviewed} ({reviewed/total*100:.1f}%)")
            print(f"Not reviewed: {not_reviewed} ({not_reviewed/total*100:.1f}%)")
            
            if reviewed > 0:
                # Count by annotation type
                not_present = len(audio_db.df.filter(pl.col("annotation") == 0))
                present = len(audio_db.df.filter(pl.col("annotation") == 1))
                uncertain = len(audio_db.df.filter(pl.col("annotation") == 3))
                
                print(f"\nAnnotation breakdown:")
                print(f"Target sound not present: {not_present} ({not_present/reviewed*100:.1f}% of reviewed)")
                print(f"Target sound present: {present} ({present/reviewed*100:.1f}% of reviewed)")
                print(f"Uncertain: {uncertain} ({uncertain/reviewed*100:.1f}% of reviewed)")
        
        with spectrogram_output:
            clear_output()
        
        with audio_info_output:
            clear_output()
        
        with status_output:
            clear_output()
    
    # Connect the buttons to their functions
    filter_button.on_click(start_annotation)
    stop_button.on_click(stop_annotation)
    
    # Connect annotation buttons
    not_present_button.on_click(lambda b: update_annotation(0))
    present_button.on_click(lambda b: update_annotation(1))
    uncertain_button.on_click(lambda b: update_annotation(3))
    
    # Create a single line title
    title = widgets.HTML("<h2 style='margin-bottom: 5px;'>Audio Clip Annotation Tool</h2>")
    
    # Create filter controls layout
    filter_section = widgets.VBox([
        widgets.HTML("<h3>Filter Audio Clips</h3>"),
        widgets.HTML("<p>Select score range to filter clips that haven't been reviewed (annotation=4):</p>"),
        score_range_slider,
        filter_button,
        filter_output
    ])
    
    # Create annotation section with right-aligned title and content
    annotation_section = widgets.VBox([
        widgets.HTML("<h3 style='text-align: right;'>Annotation Controls</h3>"),
        widgets.HTML("<p style='text-align: right;'>Listen to the clip and select the appropriate annotation:</p>"),
        annotation_buttons,
        status_output
    ], layout=widgets.Layout(
        align_items='flex-end'  # Right-align the entire section
    ))
    
    # Create a layout with the stop button at the top
    top_bar = widgets.HBox([
        title,
        widgets.HBox([stop_button], layout=widgets.Layout(justify_content='flex-end', width='100%'))
    ], layout=widgets.Layout(width='100%', margin='0px 0px 10px 0px'))  # Add margin to the bottom
    
    # Create layout for the audio display - spectrogram on left, audio player and info on right with margin
    audio_display_section = widgets.HBox([
        widgets.VBox([
            widgets.HTML("<h3>Spectrogram</h3>"),
            spectrogram_output
        ], layout=widgets.Layout(width='58%')),  # Reduced from 60% to 58% to create space
        
        # Add a small spacer for visual separation
        widgets.HTML("<div style='width: 20px;'></div>"),
        
        widgets.VBox([
            audio_info_output
        ], layout=widgets.Layout(width='38%'))   # Reduced from 40% to 38% to accommodate the spacer
    ])
    
    # Create a layout to position elements
    main_layout = widgets.VBox([
        top_bar,
        widgets.HBox([
            widgets.VBox([filter_section], layout=widgets.Layout(width='50%')),
            widgets.VBox([annotation_section], layout=widgets.Layout(width='50%'))
        ]),
        # Add a small spacing between the controls and the display
        widgets.HTML("<div style='height: 15px;'></div>"),
        audio_display_section
    ])
    
    # Display the widgets
    display(main_layout)