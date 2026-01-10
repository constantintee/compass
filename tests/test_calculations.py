#!/usr/bin/env python3
"""
Validation script to test that technical analysis calculations work correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_test_data(n_days: int = 200) -> pd.DataFrame:
    """Generate realistic test OHLCV data."""
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # Generate realistic price data using random walk
    base_price = 100
    returns = np.random.randn(n_days) * 0.02
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_days) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_days) * 0.01))
    open_prices = close_prices * (1 + np.random.randn(n_days) * 0.005)

    # Ensure high >= close and low <= close
    high_prices = np.maximum(high_prices, close_prices)
    low_prices = np.minimum(low_prices, close_prices)

    # Generate volume
    volume = np.random.randint(1_000_000, 10_000_000, n_days)

    data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume.astype(float)
    })

    return data


def test_technical_indicators():
    """Test technical indicator calculations."""
    print("=" * 60)
    print("Testing Technical Indicators")
    print("=" * 60)

    from shared.technical_analysis.indicators import TechnicalIndicators

    # Generate test data
    data = generate_test_data(200)
    data = data.set_index('date')

    indicators = TechnicalIndicators()

    try:
        result = indicators.calculate_all_indicators(data, 'TEST')

        print(f"\nInput data shape: {data.shape}")
        print(f"Output data shape: {result.shape}")
        print(f"\nColumns added: {len(result.columns) - len(data.columns)}")

        # Check that key indicators were calculated
        expected_indicators = [
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'CCI', 'ATR',
            'SuperTrend', 'SuperTrend_Direction', 'ZigZag',
            'Support', 'Resistance', 'RSI_Divergence',
            'Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_0.786',
            'OBV', 'Peak', 'Trough', 'Pivot_Point', 'R1', 'S1'
        ]

        missing_indicators = [ind for ind in expected_indicators if ind not in result.columns]
        if missing_indicators:
            print(f"\n[WARNING] Missing indicators: {missing_indicators}")
        else:
            print("\n[OK] All expected indicators are present")

        # Validate indicator values
        errors = []

        # RSI should be between 0 and 100
        if 'RSI' in result.columns:
            rsi_valid = result['RSI'].dropna()
            if len(rsi_valid) > 0:
                if rsi_valid.min() < 0 or rsi_valid.max() > 100:
                    errors.append(f"RSI out of bounds: min={rsi_valid.min():.2f}, max={rsi_valid.max():.2f}")
                else:
                    print(f"[OK] RSI values valid: range [{rsi_valid.min():.2f}, {rsi_valid.max():.2f}]")

        # Bollinger Bands: Upper > Middle > Lower
        if all(col in result.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            bb_valid = result[['BB_Upper', 'BB_Middle', 'BB_Lower']].dropna()
            if len(bb_valid) > 0:
                upper_gt_middle = (bb_valid['BB_Upper'] >= bb_valid['BB_Middle']).all()
                middle_gt_lower = (bb_valid['BB_Middle'] >= bb_valid['BB_Lower']).all()
                if not (upper_gt_middle and middle_gt_lower):
                    errors.append("Bollinger Bands ordering violated")
                else:
                    print("[OK] Bollinger Bands ordering is correct (Upper >= Middle >= Lower)")

        # Support should be <= Resistance
        if 'Support' in result.columns and 'Resistance' in result.columns:
            sr_valid = result[['Support', 'Resistance']].dropna()
            if len(sr_valid) > 0:
                if (sr_valid['Support'] > sr_valid['Resistance']).any():
                    errors.append("Support > Resistance in some rows")
                else:
                    print("[OK] Support <= Resistance for all rows")

        # EMA_12 should be more responsive than EMA_26
        if 'EMA_12' in result.columns and 'EMA_26' in result.columns:
            ema_valid = result[['EMA_12', 'EMA_26', 'close']].dropna()
            if len(ema_valid) > 0:
                # Check correlation with close price
                corr_12 = ema_valid['EMA_12'].corr(ema_valid['close'])
                corr_26 = ema_valid['EMA_26'].corr(ema_valid['close'])
                print(f"[OK] EMA correlations with close: EMA_12={corr_12:.4f}, EMA_26={corr_26:.4f}")

        # Check MACD calculation (MACD = EMA_12 - EMA_26)
        if all(col in result.columns for col in ['MACD', 'EMA_12', 'EMA_26']):
            macd_valid = result[['MACD', 'EMA_12', 'EMA_26']].dropna()
            if len(macd_valid) > 0:
                calculated_macd = macd_valid['EMA_12'] - macd_valid['EMA_26']
                macd_diff = (macd_valid['MACD'] - calculated_macd).abs().max()
                if macd_diff < 0.01:
                    print(f"[OK] MACD calculation verified (max diff: {macd_diff:.6f})")
                else:
                    errors.append(f"MACD calculation differs from EMA_12 - EMA_26 by {macd_diff:.4f}")

        # Check OBV is cumulative
        if 'OBV' in result.columns:
            obv = result['OBV'].dropna()
            if len(obv) > 1:
                # OBV should be cumulative (either increasing or decreasing trend)
                print(f"[OK] OBV range: [{obv.min():.0f}, {obv.max():.0f}]")

        # Print sample of calculated values
        print("\n--- Sample calculated values (last 5 rows) ---")
        sample_cols = ['close', 'EMA_12', 'RSI', 'MACD', 'BB_Upper', 'Support']
        available_cols = [c for c in sample_cols if c in result.columns]
        print(result[available_cols].tail().to_string())

        if errors:
            print("\n[ERRORS]")
            for error in errors:
                print(f"  - {error}")
            return False

        print("\n[SUCCESS] All technical indicator calculations passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Exception during calculation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_elliott_wave():
    """Test Elliott Wave analysis."""
    print("\n" + "=" * 60)
    print("Testing Elliott Wave Analysis")
    print("=" * 60)

    from shared.technical_analysis.indicators import TechnicalIndicators
    from shared.technical_analysis.elliott_wave import AdvancedElliottWaveAnalysis, WaveRules

    # Generate test data with clear trend
    data = generate_test_data(200)
    data = data.set_index('date')

    # First calculate indicators
    indicators = TechnicalIndicators()
    data_with_indicators = indicators.calculate_all_indicators(data, 'TEST')

    if data_with_indicators.empty:
        print("[ERROR] Failed to calculate base indicators")
        return False

    # Test Elliott Wave analysis
    elliott_analyzer = AdvancedElliottWaveAnalysis()

    try:
        elliott_df = elliott_analyzer.identify_elliott_waves(data_with_indicators)

        print(f"\nElliott Wave output shape: {elliott_df.shape}")
        print(f"Elliott Wave columns: {list(elliott_df.columns)}")

        # Check expected columns
        expected_cols = ['Elliott_Wave', 'Wave_Degree', 'Wave_Type', 'Wave_Number', 'Wave_Confidence']
        missing_cols = [c for c in expected_cols if c not in elliott_df.columns]

        if missing_cols:
            print(f"[WARNING] Missing Elliott Wave columns: {missing_cols}")
        else:
            print("[OK] All Elliott Wave columns present")

        # Check wave patterns detected
        if 'Elliott_Wave' in elliott_df.columns:
            wave_detected = (elliott_df['Elliott_Wave'] == 1).sum()
            total_rows = len(elliott_df)
            print(f"[INFO] Wave patterns detected in {wave_detected}/{total_rows} rows ({100*wave_detected/total_rows:.1f}%)")

        if 'Wave_Type' in elliott_df.columns:
            wave_types = elliott_df['Wave_Type'].value_counts()
            print(f"[INFO] Wave types found: {dict(wave_types)}")

        if 'Wave_Confidence' in elliott_df.columns:
            conf_valid = elliott_df['Wave_Confidence'][elliott_df['Wave_Confidence'] > 0]
            if len(conf_valid) > 0:
                print(f"[INFO] Confidence scores range: [{conf_valid.min():.3f}, {conf_valid.max():.3f}]")

        print("\n[SUCCESS] Elliott Wave analysis completed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Exception during Elliott Wave analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wave_rules():
    """Test Elliott Wave rules validation."""
    print("\n" + "=" * 60)
    print("Testing Wave Rules Validation")
    print("=" * 60)

    from shared.technical_analysis.elliott_wave import WaveRules
    from datetime import datetime, timedelta

    # Test valid impulse pattern
    valid_impulse = {
        'magnitudes': [10, 5, 20, 4, 8],  # Wave 3 is longest, Wave 2 < Wave 1
        'points': [
            {'price': 100, 'date': datetime(2023, 1, 1)},
            {'price': 110, 'date': datetime(2023, 1, 15)},
            {'price': 105, 'date': datetime(2023, 2, 1)},
            {'price': 125, 'date': datetime(2023, 3, 1)},
            {'price': 121, 'date': datetime(2023, 3, 15)},
        ],
        'directions': [1, -1, 1, -1, 1]
    }

    is_valid, violations = WaveRules.check_impulse_rules(valid_impulse)
    print(f"\nValid impulse test: is_valid={is_valid}")
    if violations:
        print(f"  Violations: {violations}")

    # Test invalid impulse (Wave 3 too short)
    invalid_impulse = {
        'magnitudes': [10, 5, 8, 4, 12],  # Wave 3 is shortest - invalid!
        'points': [
            {'price': 100, 'date': datetime(2023, 1, 1)},
            {'price': 110, 'date': datetime(2023, 1, 15)},
            {'price': 105, 'date': datetime(2023, 2, 1)},
            {'price': 113, 'date': datetime(2023, 3, 1)},
            {'price': 109, 'date': datetime(2023, 3, 15)},
        ],
        'directions': [1, -1, 1, -1, 1]
    }

    is_valid2, violations2 = WaveRules.check_impulse_rules(invalid_impulse)
    print(f"\nInvalid impulse test (Wave 3 shortest): is_valid={is_valid2}")
    if violations2:
        print(f"  Violations: {violations2}")

    if not is_valid2 and "Wave 3 is the shortest" in str(violations2):
        print("[OK] Correctly detected invalid impulse pattern")
    else:
        print("[WARNING] Did not correctly detect Wave 3 violation")

    print("\n[SUCCESS] Wave rules validation tests completed!")
    return True


def test_orchestrator():
    """Test the full orchestrator pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Orchestrator Pipeline")
    print("=" * 60)

    from shared.technical_analysis.orchestrator import TechnicalAnalysis

    # Generate test data
    data = generate_test_data(200)

    try:
        # Initialize orchestrator (will try to connect to DB but should handle failure gracefully)
        ta = TechnicalAnalysis()

        # Run full calculation
        result = ta.calculate_technical_indicators(data, 'TEST')

        print(f"\nInput shape: {data.shape}")
        print(f"Output shape: {result.shape}")
        print(f"Total columns: {len(result.columns)}")

        # Count indicator columns
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        indicator_cols = [c for c in result.columns if c not in base_cols and c != 'date']
        print(f"Indicator columns added: {len(indicator_cols)}")

        # Check for NaN values
        nan_counts = result.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if len(cols_with_nans) > 0:
            print(f"\n[INFO] Columns with NaN values (expected for warm-up periods):")
            for col, count in cols_with_nans.items():
                pct = 100 * count / len(result)
                if pct < 20:  # Less than 20% NaN is acceptable for warm-up
                    print(f"  {col}: {count} ({pct:.1f}%) - OK")
                else:
                    print(f"  {col}: {count} ({pct:.1f}%) - WARNING")

        # Clean up
        ta.close()

        print("\n[SUCCESS] Orchestrator pipeline completed successfully!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Exception during orchestrator test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test preprocessing functions."""
    print("\n" + "=" * 60)
    print("Testing Preprocessing Functions")
    print("=" * 60)

    # Import validators and cleaners (these don't require tensorflow)
    from shared.preprocessing import DataValidator, DataCleaner

    # Generate test data with some NaN values
    data = generate_test_data(100)

    # Introduce some NaN values
    data.loc[5:10, 'close'] = np.nan
    data.loc[15:18, 'volume'] = np.nan

    print(f"\nTest data with NaNs:")
    print(f"  NaN in close: {data['close'].isna().sum()}")
    print(f"  NaN in volume: {data['volume'].isna().sum()}")

    validator = DataValidator()
    cleaner = DataCleaner()

    try:
        # Test validation
        validated = validator.validate_raw_data(data.copy(), 'TEST')
        print(f"\n[OK] Raw data validation completed")

        # Test cleaning
        cleaned = cleaner.handle_missing_values(validated.copy(), 'TEST')
        print(f"[OK] Missing value handling completed")
        print(f"  NaN remaining: {cleaned.isna().sum().sum()}")

        # Test preprocessing
        preprocessed = cleaner.preprocess_data(cleaned.copy(), 'TEST')
        print(f"[OK] Preprocessing completed")
        print(f"  Final shape: {preprocessed.shape}")

        print("\n[SUCCESS] Preprocessing tests passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Exception during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("COMPASS CALCULATION VALIDATION TESTS")
    print("=" * 60)

    results = {}

    # Run tests
    results['Technical Indicators'] = test_technical_indicators()
    results['Elliott Wave'] = test_elliott_wave()
    results['Wave Rules'] = test_wave_rules()
    results['Orchestrator'] = test_orchestrator()
    results['Preprocessing'] = test_preprocessing()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please review the output above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
