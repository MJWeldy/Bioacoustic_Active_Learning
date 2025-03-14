import polars as pl
import numpy as np
from datetime import datetime
#from sklearn.metrics.pairwise import euclidean_distances
#import json
#import os
from typing import List, Dict, Any, Tuple

class Audio_DB:
  def __init__(self, embedding_dim: int = 1280):
      """
      Initialize the audio prediction and embedding database with polars.
      
      Args:
          embedding_dim: Dimension size of the embeddings
      """
      self.score_min = 0.0
      self.score_max = 1.0
      self.embedding_dim = embedding_dim
      self.df = pl.DataFrame(
        schema={
          'file_name': pl.Utf8,
          'file_path': pl.Utf8,
          'duration_sec': pl.Float32,
          'clip_start': pl.Float32,
          'clip_end': pl.Float32,
          'sampling_rate': pl.Int32,
          'score': pl.Float32,
          'annotation': pl.Int32,
          'created_at': pl.Datetime
        }
      )
      #'embedding': pl.List(pl.Float32),
      #'metadata': pl.Struct,  # For storing additional information
  def add_clip_row(self, 
                   file_name: str, 
                   file_path: str, 
                   duration_sec: float, 
                   clip_start: float,
                   clip_end: float,
                   sampling_rate: int) -> None:
                    #embedding: List[float],
                    #metadata: Dict[str, Any] = None) -> None:
    """
    Add an audio embedding to the database.
    
    Args:
        file_name: Unique identifier for the audio clip
        file_path: Path to the audio file
        duration_sec: Duration in seconds
        clip_start: Start time of the audio clip in seconds
        clip_end: End time of the audio clip in seconds
        sampling_rate: Audio sampling rate in Hz
        score: Predicted classifier score 0-1
        annotation: Clip annotation state[ 0: target sound not in clip,
                                           1: target sound in clip,
                                           3: reviewed but uncertain if the target sound is in the clip,
                                           4: not yet reviewed]
        NOT YET IMPLEMENTED embedding: Embedding vector for the audio clip
        NOT YET IMPLEMENTED metadata: Additional information about the clip
    """
    # Check score input 
    #if score > self.score_max or score < self.score_min:
    #   raise ValueError(f"Scores should be between {self.score_min} and {self.score_max}")
    
    # Check embedding input
    #if len(embedding) != self.embedding_dim:
    #    raise ValueError(f"Embedding dimension should be {self.embedding_dim}")
        
    #if metadata is None:
    #    metadata = {}
    
    # ensure data types
    duration_sec_float32 = np.float32(duration_sec)
    clip_start_float32 = np.float32(clip_start)
    clip_end_float32 = np.float32(clip_end)
    sampling_rate_int32 = np.int32(sampling_rate)
    score_float32 = np.float32(2.0)
    annotation_int32 = np.int32(4)
    
    
    new_row = pl.DataFrame({
        'file_name': [file_name],
        'file_path': [file_path],
        'duration_sec': [duration_sec_float32],
        'clip_start': [clip_start_float32],
        'clip_end': [clip_end_float32],
        'sampling_rate': [sampling_rate_int32],
        'score': [score_float32], #initialized at a non-real value for initial testing
        'annotation': [annotation_int32],
        'created_at': [datetime.now()]
    })

    self.df = pl.concat([self.df, new_row])

  def save_db(self, file_path: str) -> None:
    """Save the database to a file."""
    self.df.write_parquet(file_path)
  
  def load_db(self, file_path: str) -> None:
    """Load the database from a file."""
    if os.path.exists(file_path):
        self.df = pl.read_parquet(file_path)
    else:
        raise FileNotFoundError(f"Database file {file_path} not found")
  
  def populate_scores(self, scores: List[float]):
      if len(scores) != len(self.df):
        raise ValueError(f"Length of new_values ({len(scores)}) must match DataFrame length ({len(self.df)})")
      
      if any(score > self.score_max or score < self.score_min for score in scores):
        print(f"Warning: Some scores are outside the expected range [{self.score_min}, {self.score_max}]")
      
      scores_float32 = np.float32(scores)
      self.df = self.df.with_columns(pl.Series("score", scores_float32))
  
  def empty_method(self):
      pass