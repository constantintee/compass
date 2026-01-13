# shared/technical_analysis/elliott_wave.py
"""
Elliott Wave analysis implementation.

This module provides advanced Elliott Wave pattern detection and analysis.
"""

import logging
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..constants import WaveDegree, WavePatternType


class WaveRules:
    """Elliott Wave rules and guidelines validation."""

    @staticmethod
    def check_impulse_rules(pattern: Dict) -> Tuple[bool, List[str]]:
        """
        Check if pattern follows impulse wave rules.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        try:
            magnitudes = pattern.get('magnitudes', [])
            points = pattern.get('points', [])

            if len(magnitudes) < 5 or len(points) < 5:
                return False, ["Insufficient data points"]

            # Rule 1: Wave 2 never retraces more than 100% of Wave 1
            if magnitudes[1] > magnitudes[0]:
                violations.append("Wave 2 retraced more than 100% of Wave 1")

            # Rule 2: Wave 3 is never the shortest among waves 1, 3, and 5
            if magnitudes[2] <= min(magnitudes[0], magnitudes[4]):
                violations.append("Wave 3 is the shortest among waves 1, 3, and 5")

            # Rule 3: Wave 4 never overlaps with Wave 1
            wave1_high = max(points[0]['price'], points[1]['price'])
            wave4_low = min(points[3]['price'], points[4]['price'])
            if wave4_low <= wave1_high:
                violations.append("Wave 4 overlaps with Wave 1 territory")

            # Wave 3 should be at least 1.618 times Wave 1
            if magnitudes[2] < 1.618 * magnitudes[0]:
                violations.append("Wave 3 is not extended enough (should be >= 1.618 * Wave 1)")

            # Wave 5 should be at least 0.618 times Wave 1
            if magnitudes[4] < 0.618 * magnitudes[0]:
                violations.append("Wave 5 is too short (should be >= 0.618 * Wave 1)")

            # Check alternation between waves 2 and 4
            wave2_time = (points[2]['date'] - points[1]['date']).days
            wave4_time = (points[4]['date'] - points[3]['date']).days
            if wave2_time > 0 and wave4_time > 0:
                if abs(wave2_time - wave4_time) / max(wave2_time, wave4_time) < 0.382:
                    violations.append("Waves 2 and 4 lack alternation in time")

            return len(violations) == 0, violations

        except Exception as e:
            return False, [f"Error checking impulse rules: {str(e)}"]

    @staticmethod
    def check_diagonal_rules(pattern: Dict) -> Tuple[bool, List[str]]:
        """Check if pattern follows diagonal rules."""
        violations = []
        try:
            magnitudes = pattern.get('magnitudes', [])
            points = pattern.get('points', [])

            if len(magnitudes) < 5 or len(points) < 5:
                return False, ["Insufficient data points"]

            wave_lengths = [len(w) for w in pattern.get('subwaves', [[]] * 5)]
            expected_lengths = [5, 3, 5, 3, 5]

            # Check wave structure
            if wave_lengths == expected_lengths:
                wave1_width = abs(points[1]['price'] - points[0]['price'])
                wave5_width = abs(points[4]['price'] - points[3]['price'])

                if wave5_width >= wave1_width:
                    violations.append("Diagonal trendlines not converging")

                if magnitudes[2] >= magnitudes[0]:
                    violations.append("Wave 3 not shorter than Wave 1 in diagonal")
                if magnitudes[4] >= magnitudes[2]:
                    violations.append("Wave 5 not shorter than Wave 3 in diagonal")

            elif all(length == 3 for length in wave_lengths):
                if pattern.get('volume_profile') != 'declining':
                    violations.append("Volume not declining in ending diagonal")
            else:
                violations.append("Wave structure doesn't match diagonal pattern")

            return len(violations) == 0, violations

        except Exception as e:
            return False, [f"Error checking diagonal rules: {str(e)}"]

    @staticmethod
    def check_corrective_rules(pattern: Dict) -> Tuple[bool, List[str]]:
        """Check if pattern follows corrective wave rules."""
        violations = []
        try:
            magnitudes = pattern.get('magnitudes', [])
            pattern_type = pattern.get('corrective_type', '')

            if len(magnitudes) < 3:
                return False, ["Insufficient data for corrective pattern"]

            if pattern_type == 'Zigzag':
                b_retracement = magnitudes[1] / magnitudes[0] if magnitudes[0] != 0 else 0
                if not (0.5 <= b_retracement <= 0.79):
                    violations.append("B wave retracement out of range for Zigzag")

                c_extension = magnitudes[2] / magnitudes[0] if magnitudes[0] != 0 else 0
                if not (0.618 <= c_extension <= 1.618):
                    violations.append("C wave extension out of range for Zigzag")

            elif pattern_type == 'Flat':
                b_retracement = magnitudes[1] / magnitudes[0] if magnitudes[0] != 0 else 0
                if not (0.9 <= b_retracement <= 1.05):
                    violations.append("B wave retracement out of range for Flat")

                c_extension = magnitudes[2] / magnitudes[0] if magnitudes[0] != 0 else 0
                if not (1.0 <= c_extension <= 1.65):
                    violations.append("C wave extension out of range for Flat")

            elif pattern_type == 'Triangle':
                if not pattern.get('converging_trendlines', False):
                    violations.append("Triangle trendlines not converging")

                for i in range(1, len(magnitudes)):
                    if magnitudes[i] >= magnitudes[i - 1]:
                        violations.append(f"Wave {i + 1} not shorter than previous wave in Triangle")

            return len(violations) == 0, violations

        except Exception as e:
            return False, [f"Error checking corrective rules: {str(e)}"]


class AdvancedElliottWaveAnalysis:
    """Advanced Elliott Wave pattern detection and analysis."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('elliott_wave')

    def identify_elliott_waves(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to identify Elliott Wave patterns using pre-calculated indicators.

        Args:
            data: DataFrame with ZigZag, Support, and Resistance columns

        Returns:
            DataFrame with Elliott Wave pattern information
        """
        try:
            self.logger.info("Identifying Elliott waves with advanced analysis...")

            required_columns = ['ZigZag', 'Support', 'Resistance']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame(index=data.index)

            if not self._validate_input_data(data):
                return pd.DataFrame(index=data.index)

            # Get significant swing points from ZigZag
            swing_points = self._get_swing_points(data)
            if swing_points.empty:
                return pd.DataFrame(index=data.index)

            # Find all potential patterns
            wave_patterns = self._find_all_patterns(swing_points, data)

            # Create comprehensive Elliott Wave DataFrame
            return self._create_elliott_wave_dataframe(wave_patterns, data.index)

        except Exception as e:
            self.logger.error(f"Error in Elliott Wave identification: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return pd.DataFrame(index=data.index)

    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for Elliott Wave analysis."""
        try:
            required_columns = ['ZigZag', 'Support', 'Resistance', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            if data.empty:
                self.logger.error("Input data is empty")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating input data: {str(e)}")
            return False

    def _get_swing_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate swing points from ZigZag indicator."""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            mask = data['ZigZag'].notna()
            swing_points = data[mask].copy()

            if len(swing_points) < 5:
                self.logger.warning("Insufficient swing points for Elliott Wave analysis")
                return pd.DataFrame()

            # Calculate swing magnitude
            swing_points['swing_magnitude'] = np.abs(
                np.diff(swing_points['ZigZag'], prepend=swing_points['ZigZag'].iloc[0])
            )
            min_magnitude = swing_points['swing_magnitude'].mean() * 0.1

            return swing_points[swing_points['swing_magnitude'] > min_magnitude]

        except Exception as e:
            self.logger.error(f"Error processing swing points: {str(e)}")
            return pd.DataFrame()

    def _find_all_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find all possible Elliott Wave patterns."""
        all_patterns = []

        try:
            if len(swing_points) < 5:
                self.logger.warning("Insufficient swing points for pattern detection")
                return []

            min_movement = data['close'].std() * 0.5

            filtered_swings = swing_points[swing_points['swing_magnitude'] > min_movement]

            # Find different pattern types
            impulse_patterns = self._find_impulse_patterns(filtered_swings, data)
            corrective_patterns = self._find_corrective_patterns(filtered_swings, data)

            all_patterns.extend(impulse_patterns)
            all_patterns.extend(corrective_patterns)

            # Remove overlapping patterns
            all_patterns = self._remove_overlapping_patterns(all_patterns)

            # Sort by confidence score
            all_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            return all_patterns

        except Exception as e:
            self.logger.error(f"Error finding wave patterns: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return []

    def _find_impulse_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find potential impulse wave patterns."""
        patterns = []

        try:
            for i in range(len(swing_points) - 4):
                window = swing_points.iloc[i:i + 5]
                pattern = self._create_pattern_dict(window, data)

                if not pattern:
                    continue

                is_valid, _ = WaveRules.check_impulse_rules(pattern)
                if is_valid:
                    pattern['type'] = (
                        WavePatternType.IMPULSE_BULL
                        if pattern.get('trend') == 'up'
                        else WavePatternType.IMPULSE_BEAR
                    )
                    pattern['confidence'] = self._calculate_pattern_confidence(pattern)
                    pattern['degree'] = self._determine_wave_degree(pattern)
                    patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error finding impulse patterns: {str(e)}")
            return []

    def _find_corrective_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find potential corrective patterns."""
        patterns = []

        try:
            for i in range(len(swing_points) - 2):
                window = swing_points.iloc[i:i + 3]
                pattern = self._create_pattern_dict(window, data)

                if not pattern or len(pattern.get('points', [])) < 3:
                    continue

                # Check for ABC pattern
                if self._is_valid_abc_pattern(pattern):
                    pattern['type'] = WavePatternType.CORRECTION_ZIGZAG
                    pattern['confidence'] = self._calculate_pattern_confidence(pattern)
                    pattern['degree'] = self._determine_wave_degree(pattern)
                    patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error finding corrective patterns: {str(e)}")
            return []

    def _create_pattern_dict(self, window: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """Create a pattern dictionary from a window of swing points."""
        try:
            if window.empty or len(window) < 2:
                return {}

            window.index = pd.to_datetime(window.index)
            data.index = pd.to_datetime(data.index)

            pattern = {
                'points': [],
                'directions': [],
                'magnitudes': [],
                'trend': 'up' if window['ZigZag'].iloc[-1] > window['ZigZag'].iloc[0] else 'down',
                'momentum': [],
                'volume': []
            }

            prev_price = None
            prev_date = None

            for index, row in window.iterrows():
                current_price = row['ZigZag']
                current_date = index

                if prev_price is not None and prev_date is not None:
                    if pd.isna(current_price) or pd.isna(prev_price):
                        continue

                    direction = 1 if current_price > prev_price else -1
                    magnitude = abs(current_price - prev_price)

                    period_mask = (data.index >= prev_date) & (data.index <= current_date)
                    period_data = data.loc[period_mask]

                    rsi = period_data['RSI'].mean() if 'RSI' in period_data else None
                    volume = period_data['volume'].mean() if 'volume' in period_data and not period_data['volume'].empty else None

                    if magnitude > 0:
                        pattern['directions'].append(direction)
                        pattern['magnitudes'].append(magnitude)
                        pattern['momentum'].append(rsi)
                        pattern['volume'].append(volume)

                pattern['points'].append({
                    'date': current_date,
                    'price': current_price,
                    'high': row.get('high', current_price),
                    'low': row.get('low', current_price)
                })

                prev_price = current_price
                prev_date = current_date

            return pattern

        except Exception as e:
            self.logger.error(f"Error creating pattern dictionary: {str(e)}")
            return {}

    def _is_valid_abc_pattern(self, pattern: Dict) -> bool:
        """Validate ABC corrective pattern."""
        try:
            if len(pattern.get('points', [])) != 3:
                return False

            magnitudes = pattern.get('magnitudes', [])
            if len(magnitudes) < 2:
                return False

            # B should retrace 50-79% of A
            if magnitudes[0] == 0:
                return False

            b_retracement = magnitudes[1] / magnitudes[0] if len(magnitudes) > 1 else 0
            return 0.3 <= b_retracement <= 0.9

        except Exception as e:
            self.logger.error(f"Error validating ABC pattern: {str(e)}")
            return False

    def _calculate_pattern_confidence(self, pattern: Dict) -> float:
        """Calculate pattern confidence score."""
        try:
            if not pattern.get('points'):
                return 0.0

            scores = []

            # Fibonacci score
            fib_score = self._calculate_fibonacci_score(pattern)
            scores.append(fib_score * 0.3)

            # Time symmetry score
            time_score = self._calculate_time_symmetry_score(pattern)
            scores.append(time_score * 0.2)

            # Momentum score
            momentum_score = self._calculate_momentum_score(pattern)
            scores.append(momentum_score * 0.25)

            # Volume score
            volume_score = self._calculate_volume_score(pattern)
            scores.append(volume_score * 0.25)

            return round(min(sum(scores), 1.0), 3)

        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {e}")
            return 0.0

    def _calculate_fibonacci_score(self, pattern: Dict) -> float:
        """Calculate Fibonacci relationships score."""
        try:
            magnitudes = pattern.get('magnitudes', [])
            if len(magnitudes) < 2:
                return 0.0

            fibonacci_ratios = [0.236, 0.382, 0.500, 0.618, 0.786, 1.618, 2.618]
            scores = []

            for i in range(len(magnitudes) - 1):
                if magnitudes[i] == 0:
                    continue
                ratio = magnitudes[i + 1] / magnitudes[i]
                closest = min(fibonacci_ratios, key=lambda x: abs(x - ratio))
                score = 1 - min(abs(ratio - closest), 1.0)
                scores.append(score)

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci score: {e}")
            return 0.0

    def _calculate_time_symmetry_score(self, pattern: Dict) -> float:
        """Calculate time symmetry score."""
        try:
            points = pattern.get('points', [])
            if len(points) < 3:
                return 0.0

            dates = [pd.to_datetime(point['date']) for point in points]
            time_spans = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]

            if not time_spans or 0 in time_spans:
                return 0.0

            ratios = []
            for i in range(len(time_spans) - 1):
                if time_spans[i] > 0:
                    ratio = time_spans[i + 1] / time_spans[i]
                    ratios.append(min(ratio, 1 / ratio) if ratio > 0 else 0)

            return sum(ratios) / len(ratios) if ratios else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating time symmetry score: {e}")
            return 0.0

    def _calculate_momentum_score(self, pattern: Dict) -> float:
        """Calculate momentum alignment score."""
        try:
            momentum_data = pattern.get('momentum', [])
            directions = pattern.get('directions', [])

            if not momentum_data or not directions:
                return 0.5

            valid_pairs = [
                (d, m) for d, m in zip(directions, momentum_data)
                if m is not None and not np.isnan(m)
            ]

            if not valid_pairs:
                return 0.5

            scores = []
            for direction, momentum in valid_pairs:
                if direction > 0:
                    score = momentum / 100 if momentum > 50 else 0.3
                else:
                    score = (100 - momentum) / 100 if momentum < 50 else 0.3
                scores.append(score)

            return sum(scores) / len(scores) if scores else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.5

    def _calculate_volume_score(self, pattern: Dict) -> float:
        """Calculate volume confirmation score."""
        try:
            volumes = pattern.get('volume', [])
            if not volumes or len(volumes) < 2:
                return 0.5

            valid_volumes = [v for v in volumes if v is not None and v > 0]
            if len(valid_volumes) < 2:
                return 0.5

            # Check if volume supports the pattern
            x = np.arange(len(valid_volumes))
            slope, _, _, _, _ = stats.linregress(x, valid_volumes)

            # Positive slope is generally good for impulse waves
            if slope > 0:
                return min(0.5 + slope / max(valid_volumes) * 10, 1.0)
            return max(0.5 + slope / max(valid_volumes) * 10, 0.0)

        except Exception as e:
            self.logger.error(f"Error calculating volume score: {e}")
            return 0.5

    def _determine_wave_degree(self, pattern: Dict) -> str:
        """Determine the degree of the wave pattern based on time span."""
        try:
            points = pattern.get('points', [])
            if len(points) < 2:
                return WaveDegree.MINOR

            start_date = pd.to_datetime(points[0]['date'])
            end_date = pd.to_datetime(points[-1]['date'])
            time_span = (end_date - start_date).days

            if time_span > 365 * 4:
                return WaveDegree.SUPERCYCLE
            elif time_span > 365:
                return WaveDegree.CYCLE
            elif time_span > 30:
                return WaveDegree.PRIMARY
            elif time_span > 7:
                return WaveDegree.INTERMEDIATE
            else:
                return WaveDegree.MINOR

        except Exception as e:
            self.logger.error(f"Error determining wave degree: {e}")
            return WaveDegree.MINOR

    def _remove_overlapping_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Remove overlapping patterns, keeping higher confidence ones."""
        try:
            if not patterns:
                return []

            patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            non_overlapping = []
            used_dates = set()

            for pattern in patterns:
                if not pattern.get('points'):
                    continue

                start_date = pattern['points'][0]['date']
                end_date = pattern['points'][-1]['date']

                dates = set(pd.date_range(start_date, end_date, freq='D'))
                if not dates.intersection(used_dates):
                    non_overlapping.append(pattern)
                    used_dates.update(dates)

            return non_overlapping

        except Exception as e:
            self.logger.error(f"Error removing overlapping patterns: {str(e)}")
            return patterns

    def _create_elliott_wave_dataframe(self, wave_patterns: List[Dict], index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create Elliott Wave DataFrame from detected patterns."""
        try:
            n_rows = len(index)
            elliott_wave = np.zeros(n_rows)
            wave_numbers = np.zeros(n_rows)
            wave_confidence = np.zeros(n_rows)
            wave_types = np.full(n_rows, 'None', dtype='object')
            wave_degrees = np.full(n_rows, 'None', dtype='object')

            date_to_idx = pd.Series(np.arange(len(index)), index=index)

            for pattern in wave_patterns:
                if not pattern.get('points'):
                    continue

                try:
                    start_date = pd.to_datetime(pattern['points'][0]['date'])
                    end_date = pd.to_datetime(pattern['points'][-1]['date'])

                    if start_date not in date_to_idx.index or end_date not in date_to_idx.index:
                        continue

                    start_idx = date_to_idx[start_date]
                    end_idx = date_to_idx[end_date]

                    pattern_slice = slice(start_idx, end_idx + 1)
                    elliott_wave[pattern_slice] = 1
                    wave_types[pattern_slice] = pattern.get('type', 'Unknown')
                    wave_degrees[pattern_slice] = pattern.get('degree', 'Unknown')
                    wave_confidence[pattern_slice] = pattern.get('confidence', 0.0)

                    for i, point in enumerate(pattern['points'][:-1]):
                        next_point = pattern['points'][i + 1]
                        point_date = pd.to_datetime(point['date'])
                        next_date = pd.to_datetime(next_point['date'])

                        if point_date in date_to_idx.index and next_date in date_to_idx.index:
                            wave_start = date_to_idx[point_date]
                            wave_end = date_to_idx[next_date]
                            wave_numbers[wave_start:wave_end] = i + 1

                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Error processing pattern dates: {e}")
                    continue

            df = pd.DataFrame({
                'Elliott_Wave': elliott_wave,
                'Wave_Degree': wave_degrees,
                'Wave_Type': wave_types,
                'Wave_Number': wave_numbers,
                'Wave_Confidence': wave_confidence
            }, index=index)

            df.fillna({
                'Elliott_Wave': 0,
                'Wave_Degree': 'None',
                'Wave_Type': 'None',
                'Wave_Number': 0,
                'Wave_Confidence': 0.0
            }, inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error creating Elliott Wave DataFrame: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return pd.DataFrame(index=index)
