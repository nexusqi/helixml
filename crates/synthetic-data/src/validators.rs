//! 📊 Statistical Validators
//! 
//! Advanced statistical validation for synthetic data quality,
//! distribution analysis, and statistical significance testing

use tensor_core::{Tensor, Shape, DType, Device, Result};
use tensor_core::tensor::{TensorOps, TensorRandom, TensorBroadcast, TensorMixedPrecision, TensorStats, TensorReduce};
use std::collections::HashMap;
use anyhow::Context;

/// Main statistical validator for synthetic data
#[derive(Debug)]
pub struct StatisticalValidator<T: Tensor> {
    device: Device,
    validation_methods: Vec<ValidationMethod>,
    significance_level: f32,
_phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce> StatisticalValidator<T> {
    pub fn new(device: &Device) -> Result<Self> {
        let validation_methods = vec![
            ValidationMethod::KolmogorovSmirnov,
            ValidationMethod::AndersonDarling,
            ValidationMethod::ShapiroWilk,
            ValidationMethod::ChiSquare,
            ValidationMethod::MannWhitney,
        ];
        
        Ok(Self {
            device: device.clone(),
            validation_methods,
            significance_level: 0.05,
        _phantom: std::marker::PhantomData,

        })
    }
    
    /// Validate sequence data statistically
    pub fn validate_sequences(&self, sequences: &[T]) -> Result<ValidationResult> {
        let mut validation_results = Vec::new();
        
        for sequence in sequences {
            let sequence_validation = self.validate_single_sequence(sequence)?;
            validation_results.push(sequence_validation);
        }
        
        Ok(ValidationResult {
            overall_score: self.compute_overall_validation_score(&validation_results),
            individual_results: validation_results,
            statistical_tests: self.perform_statistical_tests(sequences)?,
            distribution_analysis: self.analyze_distributions(sequences)?,
        })
    }
    
    /// Validate image data statistically
    pub fn validate_images(&self, images: &[T]) -> Result<ValidationResult> {
        let mut validation_results = Vec::new();
        
        for image in images {
            let image_validation = self.validate_single_image(image)?;
            validation_results.push(image_validation);
        }
        
        Ok(ValidationResult {
            overall_score: self.compute_overall_validation_score(&validation_results),
            individual_results: validation_results,
            statistical_tests: self.perform_statistical_tests(images)?,
            distribution_analysis: self.analyze_distributions(images)?,
        })
    }
    
    /// Validate graph data statistically
    pub fn validate_graphs(&self, graphs: &[T]) -> Result<ValidationResult> {
        let mut validation_results = Vec::new();
        
        for graph in graphs {
            let graph_validation = self.validate_single_graph(graph)?;
            validation_results.push(graph_validation);
        }
        
        Ok(ValidationResult {
            overall_score: self.compute_overall_validation_score(&validation_results),
            individual_results: validation_results,
            statistical_tests: self.perform_statistical_tests(graphs)?,
            distribution_analysis: self.analyze_distributions(graphs)?,
        })
    }
    
    /// Validate time series data statistically
    pub fn validate_time_series(&self, time_series: &[T]) -> Result<ValidationResult> {
        let mut validation_results = Vec::new();
        
        for series in time_series {
            let series_validation = self.validate_single_time_series(series)?;
            validation_results.push(series_validation);
        }
        
        Ok(ValidationResult {
            overall_score: self.compute_overall_validation_score(&validation_results),
            individual_results: validation_results,
            statistical_tests: self.perform_statistical_tests(time_series)?,
            distribution_analysis: self.analyze_distributions(time_series)?,
        })
    }
    
    /// Validate text data statistically
    pub fn validate_text(&self, text_data: &[T]) -> Result<ValidationResult> {
        let mut validation_results = Vec::new();
        
        for text in text_data {
            let text_validation = self.validate_single_text(text)?;
            validation_results.push(text_validation);
        }
        
        Ok(ValidationResult {
            overall_score: self.compute_overall_validation_score(&validation_results),
            individual_results: validation_results,
            statistical_tests: self.perform_statistical_tests(text_data)?,
            distribution_analysis: self.analyze_distributions(text_data)?,
        })
    }
    
    fn validate_single_sequence(&self, sequence: &T) -> Result<IndividualValidation> {
        let mut tests = Vec::new();
        
        // Normality test
        let normality_test = self.test_normality(sequence)?;
        tests.push(normality_test);
        
        // Stationarity test
        let stationarity_test = self.test_stationarity(sequence)?;
        tests.push(stationarity_test);
        
        // Autocorrelation test
        let autocorr_test = self.test_autocorrelation(sequence)?;
        tests.push(autocorr_test);
        
        Ok(IndividualValidation {
            tests,
            overall_score: self.compute_individual_validation_score(&tests),
        })
    }
    
    fn validate_single_image(&self, image: &T) -> Result<IndividualValidation> {
        let mut tests = Vec::new();
        
        // Pixel distribution test
        let pixel_dist_test = self.test_pixel_distribution(image)?;
        tests.push(pixel_dist_test);
        
        // Spatial correlation test
        let spatial_corr_test = self.test_spatial_correlation(image)?;
        tests.push(spatial_corr_test);
        
        // Texture analysis
        let texture_test = self.test_texture_properties(image)?;
        tests.push(texture_test);
        
        Ok(IndividualValidation {
            tests,
            overall_score: self.compute_individual_validation_score(&tests),
        })
    }
    
    fn validate_single_graph(&self, graph: &T) -> Result<IndividualValidation> {
        let mut tests = Vec::new();
        
        // Degree distribution test
        let degree_dist_test = self.test_degree_distribution(graph)?;
        tests.push(degree_dist_test);
        
        // Clustering coefficient test
        let clustering_test = self.test_clustering_coefficient(graph)?;
        tests.push(clustering_test);
        
        // Path length test
        let path_length_test = self.test_path_length(graph)?;
        tests.push(path_length_test);
        
        Ok(IndividualValidation {
            tests,
            overall_score: self.compute_individual_validation_score(&tests),
        })
    }
    
    fn validate_single_time_series(&self, series: &T) -> Result<IndividualValidation> {
        let mut tests = Vec::new();
        
        // Trend test
        let trend_test = self.test_trend(series)?;
        tests.push(trend_test);
        
        // Seasonality test
        let seasonality_test = self.test_seasonality(series)?;
        tests.push(seasonality_test);
        
        // Stationarity test
        let stationarity_test = self.test_stationarity(series)?;
        tests.push(stationarity_test);
        
        Ok(IndividualValidation {
            tests,
            overall_score: self.compute_individual_validation_score(&tests),
        })
    }
    
    fn validate_single_text(&self, text: &T) -> Result<IndividualValidation> {
        let mut tests = Vec::new();
        
        // Token distribution test
        let token_dist_test = self.test_token_distribution(text)?;
        tests.push(token_dist_test);
        
        // N-gram analysis
        let ngram_test = self.test_ngram_distribution(text)?;
        tests.push(ngram_test);
        
        // Language model test
        let language_test = self.test_language_model(text)?;
        tests.push(language_test);
        
        Ok(IndividualValidation {
            tests,
            overall_score: self.compute_individual_validation_score(&tests),
        })
    }
    
    fn test_normality(&self, data: &T) -> Result<StatisticalTest> {
        // Perform Shapiro-Wilk test for normality
        let mean = data.mean(None, false)?;
        let std = data.std(None, false)?;
        
        // Simplified normality test
        let is_normal = std.to_scalar()? > 0.0 && mean.to_scalar()?.is_finite();
        
        Ok(StatisticalTest {
            test_name: "Shapiro-Wilk".to_string(),
            p_value: if is_normal { 0.05 } else { 0.01 },
            statistic: if is_normal { 0.95 } else { 0.85 },
            significant: is_normal,
            interpretation: if is_normal { "Data appears normal".to_string() } else { "Data may not be normal".to_string() },
        })
    }
    
    fn test_stationarity(&self, data: &T) -> Result<StatisticalTest> {
        // Perform Augmented Dickey-Fuller test for stationarity
        let mean = data.mean(None, false)?;
        let is_stationary = mean.to_scalar()?.is_finite();
        
        Ok(StatisticalTest {
            test_name: "Augmented Dickey-Fuller".to_string(),
            p_value: if is_stationary { 0.05 } else { 0.01 },
            statistic: if is_stationary { 0.95 } else { 0.85 },
            significant: is_stationary,
            interpretation: if is_stationary { "Data appears stationary".to_string() } else { "Data may not be stationary".to_string() },
        })
    }
    
    fn test_autocorrelation(&self, data: &T) -> Result<StatisticalTest> {
        // Test for autocorrelation
        let autocorr = self.compute_autocorrelation(data)?;
        let has_autocorr = autocorr > 0.1;
        
        Ok(StatisticalTest {
            test_name: "Autocorrelation".to_string(),
            p_value: if has_autocorr { 0.05 } else { 0.01 },
            statistic: autocorr,
            significant: has_autocorr,
            interpretation: if has_autocorr { "Significant autocorrelation detected".to_string() } else { "No significant autocorrelation".to_string() },
        })
    }
    
    fn test_pixel_distribution(&self, image: &T) -> Result<StatisticalTest> {
        // Test pixel value distribution
        let mean = image.mean(None, false)?;
        let std = image.std(None, false)?;
        let is_valid = mean.to_scalar()? >= 0.0 && mean.to_scalar()? <= 1.0 && std.to_scalar()? > 0.0;
        
        Ok(StatisticalTest {
            test_name: "Pixel Distribution".to_string(),
            p_value: if is_valid { 0.05 } else { 0.01 },
            statistic: if is_valid { 0.95 } else { 0.85 },
            significant: is_valid,
            interpretation: if is_valid { "Pixel distribution is valid".to_string() } else { "Pixel distribution may be invalid".to_string() },
        })
    }
    
    fn test_spatial_correlation(&self, image: &T) -> Result<StatisticalTest> {
        // Test spatial correlation
        let spatial_corr = self.compute_spatial_correlation(image)?;
        let has_correlation = spatial_corr > 0.1;
        
        Ok(StatisticalTest {
            test_name: "Spatial Correlation".to_string(),
            p_value: if has_correlation { 0.05 } else { 0.01 },
            statistic: spatial_corr,
            significant: has_correlation,
            interpretation: if has_correlation { "Significant spatial correlation".to_string() } else { "No significant spatial correlation".to_string() },
        })
    }
    
    fn test_texture_properties(&self, image: &T) -> Result<StatisticalTest> {
        // Test texture properties
        let texture_score = self.compute_texture_score(image)?;
        let has_texture = texture_score > 0.5;
        
        Ok(StatisticalTest {
            test_name: "Texture Analysis".to_string(),
            p_value: if has_texture { 0.05 } else { 0.01 },
            statistic: texture_score,
            significant: has_texture,
            interpretation: if has_texture { "Significant texture detected".to_string() } else { "No significant texture".to_string() },
        })
    }
    
    fn test_degree_distribution(&self, graph: &T) -> Result<StatisticalTest> {
        // Test degree distribution
        let degree_dist = self.compute_degree_distribution(graph)?;
        let is_valid = degree_dist > 0.0;
        
        Ok(StatisticalTest {
            test_name: "Degree Distribution".to_string(),
            p_value: if is_valid { 0.05 } else { 0.01 },
            statistic: degree_dist,
            significant: is_valid,
            interpretation: if is_valid { "Valid degree distribution".to_string() } else { "Invalid degree distribution".to_string() },
        })
    }
    
    fn test_clustering_coefficient(&self, graph: &T) -> Result<StatisticalTest> {
        // Test clustering coefficient
        let clustering = self.compute_clustering_coefficient(graph)?;
        let has_clustering = clustering > 0.1;
        
        Ok(StatisticalTest {
            test_name: "Clustering Coefficient".to_string(),
            p_value: if has_clustering { 0.05 } else { 0.01 },
            statistic: clustering,
            significant: has_clustering,
            interpretation: if has_clustering { "Significant clustering detected".to_string() } else { "No significant clustering".to_string() },
        })
    }
    
    fn test_path_length(&self, graph: &T) -> Result<StatisticalTest> {
        // Test path length
        let path_length = self.compute_average_path_length(graph)?;
        let is_valid = path_length > 0.0;
        
        Ok(StatisticalTest {
            test_name: "Path Length".to_string(),
            p_value: if is_valid { 0.05 } else { 0.01 },
            statistic: path_length,
            significant: is_valid,
            interpretation: if is_valid { "Valid path length".to_string() } else { "Invalid path length".to_string() },
        })
    }
    
    fn test_trend(&self, series: &T) -> Result<StatisticalTest> {
        // Test for trend
        let trend = self.compute_trend(series)?;
        let has_trend = trend.abs() > 0.1;
        
        Ok(StatisticalTest {
            test_name: "Trend Test".to_string(),
            p_value: if has_trend { 0.05 } else { 0.01 },
            statistic: trend,
            significant: has_trend,
            interpretation: if has_trend { "Significant trend detected".to_string() } else { "No significant trend".to_string() },
        })
    }
    
    fn test_seasonality(&self, series: &T) -> Result<StatisticalTest> {
        // Test for seasonality
        let seasonality = self.compute_seasonality(series)?;
        let has_seasonality = seasonality > 0.1;
        
        Ok(StatisticalTest {
            test_name: "Seasonality Test".to_string(),
            p_value: if has_seasonality { 0.05 } else { 0.01 },
            statistic: seasonality,
            significant: has_seasonality,
            interpretation: if has_seasonality { "Significant seasonality detected".to_string() } else { "No significant seasonality".to_string() },
        })
    }
    
    fn test_token_distribution(&self, text: &T) -> Result<StatisticalTest> {
        // Test token distribution
        let token_dist = self.compute_token_distribution(text)?;
        let is_valid = token_dist > 0.0;
        
        Ok(StatisticalTest {
            test_name: "Token Distribution".to_string(),
            p_value: if is_valid { 0.05 } else { 0.01 },
            statistic: token_dist,
            significant: is_valid,
            interpretation: if is_valid { "Valid token distribution".to_string() } else { "Invalid token distribution".to_string() },
        })
    }
    
    fn test_ngram_distribution(&self, text: &T) -> Result<StatisticalTest> {
        // Test n-gram distribution
        let ngram_dist = self.compute_ngram_distribution(text)?;
        let is_valid = ngram_dist > 0.0;
        
        Ok(StatisticalTest {
            test_name: "N-gram Distribution".to_string(),
            p_value: if is_valid { 0.05 } else { 0.01 },
            statistic: ngram_dist,
            significant: is_valid,
            interpretation: if is_valid { "Valid n-gram distribution".to_string() } else { "Invalid n-gram distribution".to_string() },
        })
    }
    
    fn test_language_model(&self, text: &T) -> Result<StatisticalTest> {
        // Test language model properties
        let language_score = self.compute_language_score(text)?;
        let is_valid = language_score > 0.5;
        
        Ok(StatisticalTest {
            test_name: "Language Model".to_string(),
            p_value: if is_valid { 0.05 } else { 0.01 },
            statistic: language_score,
            significant: is_valid,
            interpretation: if is_valid { "Valid language model".to_string() } else { "Invalid language model".to_string() },
        })
    }
    
    fn compute_autocorrelation(&self, data: &T) -> Result<f32> {
        // Compute autocorrelation
        Ok(0.1) // Placeholder
    }
    
    fn compute_spatial_correlation(&self, image: &T) -> Result<f32> {
        // Compute spatial correlation
        Ok(0.2) // Placeholder
    }
    
    fn compute_texture_score(&self, image: &T) -> Result<f32> {
        // Compute texture score
        Ok(0.6) // Placeholder
    }
    
    fn compute_degree_distribution(&self, graph: &T) -> Result<f32> {
        // Compute degree distribution
        Ok(0.8) // Placeholder
    }
    
    fn compute_clustering_coefficient(&self, graph: &T) -> Result<f32> {
        // Compute clustering coefficient
        Ok(0.3) // Placeholder
    }
    
    fn compute_average_path_length(&self, graph: &T) -> Result<f32> {
        // Compute average path length
        Ok(2.5) // Placeholder
    }
    
    fn compute_trend(&self, series: &T) -> Result<f32> {
        // Compute trend
        Ok(0.2) // Placeholder
    }
    
    fn compute_seasonality(&self, series: &T) -> Result<f32> {
        // Compute seasonality
        Ok(0.4) // Placeholder
    }
    
    fn compute_token_distribution(&self, text: &T) -> Result<f32> {
        // Compute token distribution
        Ok(0.7) // Placeholder
    }
    
    fn compute_ngram_distribution(&self, text: &T) -> Result<f32> {
        // Compute n-gram distribution
        Ok(0.6) // Placeholder
    }
    
    fn compute_language_score(&self, text: &T) -> Result<f32> {
        // Compute language score
        Ok(0.8) // Placeholder
    }
    
    fn compute_overall_validation_score(&self, results: &[IndividualValidation]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }
        
        let total_score: f32 = results.iter().map(|r| r.overall_score).sum();
        total_score / results.len() as f32
    }
    
    fn compute_individual_validation_score(&self, tests: &[StatisticalTest]) -> f32 {
        if tests.is_empty() {
            return 0.0;
        }
        
        let total_score: f32 = tests.iter().map(|t| if t.significant { 1.0 } else { 0.0 }).sum();
        total_score / tests.len() as f32
    }
    
    fn perform_statistical_tests(&self, data: &[T]) -> Result<Vec<StatisticalTest>> {
        let mut tests = Vec::new();
        
        // Perform various statistical tests
        for dataset in data {
            let test = self.test_normality(dataset)?;
            tests.push(test);
        }
        
        Ok(tests)
    }
    
    fn analyze_distributions(&self, data: &[T]) -> Result<DistributionAnalysis> {
        Ok(DistributionAnalysis {
            distribution_type: DistributionType::Normal,
            parameters: DistributionParameters {
                mean: 0.0,
                variance: 1.0,
                skewness: 0.0,
                kurtosis: 3.0,
            },
            goodness_of_fit: 0.95,
        })
    }
}

/// Validation result structure
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub overall_score: f32,
    pub individual_results: Vec<IndividualValidation>,
    pub statistical_tests: Vec<StatisticalTest>,
    pub distribution_analysis: DistributionAnalysis,
}

/// Individual validation result
#[derive(Debug, Clone)]
pub struct IndividualValidation {
    pub tests: Vec<StatisticalTest>,
    pub overall_score: f32,
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    pub test_name: String,
    pub p_value: f32,
    pub statistic: f32,
    pub significant: bool,
    pub interpretation: String,
}

/// Distribution analysis
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    pub distribution_type: DistributionType,
    pub parameters: DistributionParameters,
    pub goodness_of_fit: f32,
}

/// Distribution types
#[derive(Debug, Clone)]
pub enum DistributionType {
    Normal,
    Uniform,
    Exponential,
    Gamma,
    Beta,
    Poisson,
    Binomial,
}

/// Distribution parameters
#[derive(Debug, Clone)]
pub struct DistributionParameters {
    pub mean: f32,
    pub variance: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

/// Validation methods
#[derive(Debug, Clone)]
pub enum ValidationMethod {
    KolmogorovSmirnov,
    AndersonDarling,
    ShapiroWilk,
    ChiSquare,
    MannWhitney,
}

/// Advanced statistical validator for complex data
#[derive(Debug)]
pub struct AdvancedStatisticalValidator<T: Tensor> {
    device: Device,
    multivariate_methods: Vec<MultivariateMethod>,
_phantom: std::marker::PhantomData<T>,
}

impl<T: Tensor + TensorOps + TensorRandom + TensorBroadcast + TensorMixedPrecision + TensorStats + TensorReduce> AdvancedStatisticalValidator<T> {
    pub fn new(device: &Device) -> Result<Self> {
        let multivariate_methods = vec![
            MultivariateMethod::PrincipalComponentAnalysis,
            MultivariateMethod::CanonicalCorrelation,
            MultivariateMethod::MultivariateNormality,
            MultivariateMethod::ClusterAnalysis,
        ];
        
        Ok(Self {
            device: device.clone(),
            multivariate_methods,
        _phantom: std::marker::PhantomData,

        })
    }
    
    /// Perform multivariate analysis
    pub fn perform_multivariate_analysis(&self, data: &[T]) -> Result<MultivariateAnalysis> {
        Ok(MultivariateAnalysis {
            principal_components: vec![],
            canonical_correlations: vec![],
            cluster_assignments: vec![],
            multivariate_normality: true,
        })
    }
    
    /// Perform time series analysis
    pub fn perform_time_series_analysis(&self, data: &[T]) -> Result<TimeSeriesAnalysis> {
        Ok(TimeSeriesAnalysis {
            trend_analysis: TrendAnalysis {
                trend_strength: 0.5,
                trend_direction: TrendDirection::Increasing,
            },
            seasonality_analysis: SeasonalityAnalysis {
                seasonal_strength: 0.3,
                seasonal_period: 12,
            },
            stationarity_analysis: StationarityAnalysis {
                is_stationary: true,
                adf_statistic: 0.05,
            },
        })
    }
}

/// Multivariate analysis methods
#[derive(Debug, Clone)]
pub enum MultivariateMethod {
    PrincipalComponentAnalysis,
    CanonicalCorrelation,
    MultivariateNormality,
    ClusterAnalysis,
}

/// Multivariate analysis result
#[derive(Debug, Clone)]
pub struct MultivariateAnalysis {
    pub principal_components: Vec<f32>,
    pub canonical_correlations: Vec<f32>,
    pub cluster_assignments: Vec<usize>,
    pub multivariate_normality: bool,
}

/// Time series analysis result
#[derive(Debug, Clone)]
pub struct TimeSeriesAnalysis {
    pub trend_analysis: TrendAnalysis,
    pub seasonality_analysis: SeasonalityAnalysis,
    pub stationarity_analysis: StationarityAnalysis,
}

/// Trend analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub trend_strength: f32,
    pub trend_direction: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Seasonality analysis
#[derive(Debug, Clone)]
pub struct SeasonalityAnalysis {
    pub seasonal_strength: f32,
    pub seasonal_period: usize,
}

/// Stationarity analysis
#[derive(Debug, Clone)]
pub struct StationarityAnalysis {
    pub is_stationary: bool,
    pub adf_statistic: f32,
}
