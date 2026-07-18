import React, { useState, useEffect, useRef } from "react";
import {
  UploadCloud,
  FileText,
  Settings,
  Play,
  CheckCircle,
  AlertTriangle,
  Cpu,
  TrendingUp,
  Sparkles,
  Code,
  Download,
  ChevronRight,
  Copy,
  Check,
  Eye,
  RefreshCw,
  Info,
  Lock,
  User,
  LogIn,
  LogOut
} from "lucide-react";

// Helper to safely parse JSON response and avoid "Unexpected end of JSON input"
const safeParseJson = async (response) => {
  try {
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      return await response.json();
    }
  } catch (e) {
    console.error("JSON parsing error:", e);
  }
  return null;
};

export default function App() {
  // Authentication states
  const [token, setToken] = useState(localStorage.getItem("mlforge_token") || "");
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState("login"); // login, signup
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authConfirmPassword, setAuthConfirmPassword] = useState("");
  const [authError, setAuthError] = useState(null);
  const [authLoading, setAuthLoading] = useState(false);

  // Workflow step state (0: Dashboard, 1: Upload, 2: Configure, 3: Leaderboard / Training, 4: Export)
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Polling interval ref for cleanup
  const pollingIntervalRef = useRef(null);

  // Dashboard states
  const [userDatasets, setUserDatasets] = useState([]);
  const [userRuns, setUserRuns] = useState([]);
  const [dashboardLoading, setDashboardLoading] = useState(false);

  // Dataset states
  const [datasetId, setDatasetId] = useState(null);
  const [datasetName, setDatasetName] = useState("");
  const [totalRows, setTotalRows] = useState(0);
  const [totalCols, setTotalCols] = useState(0);
  const [sampleData, setSampleData] = useState([]);
  const [columnProfile, setColumnProfile] = useState({});
  const [activeProfileTab, setActiveProfileTab] = useState("preview");

  // AutoML config states
  const [targetColumn, setTargetColumn] = useState("");
  const [problemType, setProblemType] = useState("classification");
  const [scaling, setScaling] = useState("standard");
  const [imputation, setImputation] = useState("median");
  const [categoricalImputation, setCategoricalImputation] = useState("most_frequent");
  const [categoricalEncoding, setCategoricalEncoding] = useState("onehot");
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [activePreviewFeature, setActivePreviewFeature] = useState("");
  const [previewData, setPreviewData] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [selectedModelsToTrain, setSelectedModelsToTrain] = useState([]);

  // Run training states
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState("pending");
  const [leaderboard, setLeaderboard] = useState([]);
  const [bestModelName, setBestModelName] = useState("");
  const [selectedModelIndex, setSelectedModelIndex] = useState(0);
  const [runError, setRunError] = useState(null);

  // Saved models & comparison states
  const [comparedModelNames, setComparedModelNames] = useState([]);
  const [rightPanelTab, setRightPanelTab] = useState("insights"); // "insights", "comparison"
  
  // Dashboard saved models list
  const [savedModels, setSavedModels] = useState([]);
  const [savedModelsLoading, setSavedModelsLoading] = useState(false);
  
  // Model saving modal state
  const [showSaveModalFor, setShowSaveModalFor] = useState(null);
  const [saveCustomName, setSaveCustomName] = useState("");
  const [saveModelLoading, setSaveModelLoading] = useState(false);
  const [saveModelError, setSaveModelError] = useState(null);
  
  // Active test predictor model state
  const [activeTestingModel, setActiveTestingModel] = useState(null);
  const [testingModelDetails, setTestingModelDetails] = useState(null);
  const [testingModelLoading, setTestingModelLoading] = useState(false);
  const [predictionInputs, setPredictionInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  // Deployment states
  const [exportCode, setExportCode] = useState("");
  const [exportInstructions, setExportInstructions] = useState("");
  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedInstructions, setCopiedInstructions] = useState(false);

  const fileInputRef = useRef(null);

  // Auth fetch user profile effect
  useEffect(() => {
    if (token) {
      fetchUserProfile(token);
    }
  }, [token]);

  // Default to selecting all models when problemType changes
  useEffect(() => {
    let defaultModels = [];
    if (problemType === "classification") {
      defaultModels = [
        "Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Neural Network",
        "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors", 
        "Gradient Boosting", "AdaBoost", "Extra Trees", "Naive Bayes"
      ];
    } else if (problemType === "regression") {
      defaultModels = [
        "Linear Regression", "Random Forest", "XGBoost", "LightGBM", "Neural Network",
        "Support Vector Regressor", "Decision Tree", "K-Nearest Neighbors",
        "Gradient Boosting", "AdaBoost", "Extra Trees", "Ridge Regression", "Lasso Regression"
      ];
    } else if (problemType === "clustering") {
      defaultModels = [
        "K-Means", "Birch", "Mean Shift", "Affinity Propagation", "Gaussian Mixture", "Mini Batch K-Means"
      ];
    }
    setSelectedModelsToTrain(defaultModels);
  }, [problemType]);

  // Cleanup polling interval on component unmount and when step changes
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, []);

  // Clear polling intervals when leaving the leaderboard step
  useEffect(() => {
    if (step !== 3 && pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }, [step]);

  const fetchUserProfile = async (authToken) => {
    try {
      const response = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (response.ok) {
        const data = await response.json();
        setUser(data);
        fetchDashboardData(authToken);
      } else {
        handleLogout();
      }
    } catch (err) {
      console.error("Profile load failed:", err);
      handleLogout();
    }
  };

  const fetchDashboardData = async (authToken) => {
    setDashboardLoading(true);
    try {
      const [dsRes, runsRes, savedRes] = await Promise.all([
        fetch("/api/datasets", { headers: { Authorization: `Bearer ${authToken}` } }),
        fetch("/api/runs", { headers: { Authorization: `Bearer ${authToken}` } }),
        fetch("/api/saved-models", { headers: { Authorization: `Bearer ${authToken}` } })
      ]);
      if (dsRes.ok && runsRes.ok && savedRes.ok) {
        const datasets = await dsRes.json();
        const runs = await runsRes.json();
        const saved = await savedRes.json();
        setUserDatasets(datasets);
        setUserRuns(runs);
        setSavedModels(saved);
      }
    } catch (err) {
      console.error("Failed to load dashboard data:", err);
    } finally {
      setDashboardLoading(false);
    }
  };

  const navigateToDashboard = () => {
    setStep(0);
    if (token) {
      fetchDashboardData(token);
    }
  };

  const handleLogout = () => {
    // Clear any polling intervals
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    localStorage.removeItem("mlforge_token");
    setToken("");
    setUser(null);
    setStep(0);
    // Reset dataset and run states
    setDatasetId(null);
    setDatasetName("");
    setTotalRows(0);
    setTotalCols(0);
    setSampleData([]);
    setColumnProfile({});
    setRunId(null);
    setRunStatus("pending");
    setLeaderboard([]);
    setBestModelName("");
    setUserDatasets([]);
    setUserRuns([]);
    setSavedModels([]);
    setComparedModelNames([]);
    setRightPanelTab("insights");
    setShowSaveModalFor(null);
    setActiveTestingModel(null);
    setTestingModelDetails(null);
    setPredictionResult(null);
  };

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setAuthError(null);
    
    if (!authUsername || !authPassword) {
      setAuthError("Please fill in all fields.");
      return;
    }

    if (authMode === "signup" && authPassword !== authConfirmPassword) {
      setAuthError("Passwords do not match.");
      return;
    }

    setAuthLoading(true);
    try {
      if (authMode === "signup") {
        const res = await fetch("/api/auth/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: authUsername, password: authPassword }),
        });
        
        if (!res.ok) {
          const data = await safeParseJson(res);
          let errorMessage = "Signup failed.";
          if (data) {
            if (typeof data.detail === "string") {
              errorMessage = data.detail;
            } else if (Array.isArray(data.detail)) {
              errorMessage = data.detail.map(err => err.msg).join(", ");
            }
          } else {
            const text = await res.text().catch(() => "");
            if (res.status === 504 || res.status === 502) {
              errorMessage = "Backend connection timeout. Is the backend server running?";
            } else if (res.status === 503) {
              errorMessage = text || "Database service unavailable. Please check if MongoDB is running.";
            } else {
              errorMessage = `Server error (${res.status}): ${text.substring(0, 100) || res.statusText}`;
            }
          }
          throw new Error(errorMessage);
        }
        
        // Auto-login
        const loginRes = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: authUsername, password: authPassword }),
        });
        
        if (!loginRes.ok) {
          throw new Error("Signup succeeded but login failed. Please login manually.");
        }
        
        const loginData = await safeParseJson(loginRes);
        if (!loginData || !loginData.access_token) {
          throw new Error("Signup succeeded but login failed to receive access token. Please login manually.");
        }
        localStorage.setItem("mlforge_token", loginData.access_token);
        setToken(loginData.access_token);
      } else {
        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: authUsername, password: authPassword }),
        });
        
        if (!res.ok) {
          const data = await safeParseJson(res);
          let errorMessage = "Incorrect username or password.";
          if (data) {
            if (typeof data.detail === "string") {
              errorMessage = data.detail;
            } else if (Array.isArray(data.detail)) {
              errorMessage = data.detail.map(err => err.msg).join(", ");
            }
          } else {
            const text = await res.text().catch(() => "");
            if (res.status === 504 || res.status === 502) {
              errorMessage = "Backend connection timeout. Is the backend server running?";
            } else if (res.status === 503) {
              errorMessage = text || "Database service unavailable. Please check if MongoDB is running.";
            } else {
              errorMessage = `Server error (${res.status}): ${text.substring(0, 100) || res.statusText}`;
            }
          }
          throw new Error(errorMessage);
        }
        
        const data = await safeParseJson(res);
        if (!data || !data.access_token) {
          throw new Error("Failed to receive access token. Please try again.");
        }
        localStorage.setItem("mlforge_token", data.access_token);
        setToken(data.access_token);
      }
      
      // Clear fields
      setAuthUsername("");
      setAuthPassword("");
      setAuthConfirmPassword("");
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleDownloadModel = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch(`/api/runs/${runId}/models/${bestModelName}/download`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      if (!response.ok) throw new Error("Failed to download model file.");
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `mlforge_pipeline_${runId}.pkl`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadDatasetIntoWorkspace = async (dsId, goToStep = 1) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/datasets/${dsId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Failed to load dataset.");
      }
      const data = await response.json();
      setDatasetId(data.id);
      setDatasetName(data.name);
      setTotalRows(data.total_rows);
      setTotalCols(data.total_cols);
      setSampleData(data.sample || []);
      setColumnProfile(data.profile || {});
      setTargetColumn(data.auto_target || "");
      setProblemType(data.auto_problem_type || "classification");
      setActivePreviewFeature("");
      setPreviewData(null);
      setPreviewError(null);
      setRunId(null);
      setRunStatus("pending");
      setLeaderboard([]);
      setBestModelName("");
      setStep(goToStep);
      setActiveProfileTab("preview");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Auto-select features when target column changes
  useEffect(() => {
    if (columnProfile && targetColumn) {
      // Select all columns except the target and columns identified as ID or Constant
      const initialFeatures = Object.keys(columnProfile).filter(
        (col) =>
          col !== targetColumn &&
          columnProfile[col].type !== "id" &&
          columnProfile[col].type !== "constant"
      );
      setSelectedFeatures(initialFeatures);
    }
  }, [targetColumn, columnProfile]);

  // Auto-select first previewable feature for any loaded dataset
  useEffect(() => {
    if (step !== 2 || !columnProfile) return;

    const previewable = Object.keys(columnProfile).filter(
      (col) => col !== targetColumn && columnProfile[col].type !== "constant"
    );
    if (previewable.length === 0) return;

    setActivePreviewFeature((prev) => {
      if (prev && previewable.includes(prev)) return prev;
      const preferred = previewable.find((col) => selectedFeatures.includes(col));
      return preferred || previewable[0];
    });
  }, [step, columnProfile, targetColumn, selectedFeatures]);

  // Fetch live preprocessing preview when feature or config changes
  useEffect(() => {
    if (!datasetId || !activePreviewFeature || step !== 2) {
      return;
    }

    const fetchPreview = async () => {
      setPreviewLoading(true);
      setPreviewError(null);
      try {
        const response = await fetch(`/api/datasets/${datasetId}/preview_feature`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            feature: activePreviewFeature,
            numeric_imputation: imputation,
            scaling: scaling,
            categorical_imputation: categoricalImputation,
            categorical_encoding: categoricalEncoding,
          }),
        });
        if (!response.ok) {
          const errDetail = await response.json();
          throw new Error(errDetail.detail || "Failed to load preview.");
        }
        const data = await response.json();
        setPreviewData(data);
      } catch (err) {
        setPreviewError(err.message);
        setPreviewData(null);
      } finally {
        setPreviewLoading(false);
      }
    };

    fetchPreview();
  }, [datasetId, activePreviewFeature, imputation, scaling, categoricalImputation, categoricalEncoding, step, token]);

  // Handle file drop / upload
  const handleFileUpload = async (file) => {
    if (!file) return;
    if (!file.name.endsWith(".csv")) {
      setError("Only CSV files are supported.");
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
        headers: {
          Authorization: `Bearer ${token}`
        }
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Failed to upload and profile dataset.");
      }

      const data = await response.json();
      setDatasetId(data.dataset_id);
      setDatasetName(data.name);
      setTotalRows(data.total_rows);
      setTotalCols(data.total_cols);
      setSampleData(data.sample);
      setColumnProfile(data.profile);
      setTargetColumn(data.auto_target);
      setProblemType(data.auto_problem_type);
      setActivePreviewFeature("");
      setPreviewData(null);
      setPreviewError(null);
      setStep(1); // Set or refresh upload page status
      setActiveProfileTab("preview");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  // Submit training request to background thread
  const startTraining = async () => {
    if (problemType !== "clustering" && !targetColumn) {
      setError("Please select a target column first.");
      return;
    }
    if (selectedFeatures.length === 0) {
      setError("Please select at least one training feature.");
      return;
    }
    if (selectedModelsToTrain.length === 0) {
      setError("Please select at least one ML model to train.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          dataset_id: datasetId,
          target: problemType === "clustering" ? "" : targetColumn,
          features: selectedFeatures,
          problem_type: problemType,
          scaling: scaling,
          imputation: imputation,
          categorical_imputation: categoricalImputation,
          categorical_encoding: categoricalEncoding,
          selected_models: selectedModelsToTrain,
        }),
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Failed to start training.");
      }

      const data = await response.json();
      setRunId(data.id);
      setRunStatus(data.status);
      setStep(3); // Go to training status
      pollTrainingStatus(data.id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Poll training progress
  const pollTrainingStatus = (id) => {
    // Clear any existing polling intervals to prevent duplicates
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/runs/${id}`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        if (response.status === 401) {
          clearInterval(interval);
          pollingIntervalRef.current = null;
          handleLogout();
          return;
        }
        if (!response.ok) return;

        const data = await response.json();
        setRunStatus(data.status);
        setLeaderboard(data.leaderboard || []);
        setBestModelName(data.best_model || "");
        setRunError(data.error || null);

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
          pollingIntervalRef.current = null;
          setSelectedModelIndex(0); // Select the top model by default
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 2000);

    // Store the interval ID for cleanup
    pollingIntervalRef.current = interval;
  };

  // Retrieve code export instructions
  const loadExportCode = async () => {
    if (!runId || !bestModelName) return;
    setLoading(true);
    try {
      const response = await fetch(`/api/runs/${runId}/models/${bestModelName}/export-code`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      if (!response.ok) throw new Error("Failed to fetch export configurations.");
      const data = await response.json();
      setExportCode(data.code);
      setExportInstructions(data.instructions);
      setStep(4); // Move to Deploy screen
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text, type) => {
    navigator.clipboard.writeText(text);
    if (type === "code") {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    } else {
      setCopiedInstructions(true);
      setTimeout(() => setCopiedInstructions(false), 2000);
    }
  };

  const confirmSaveModel = async (e) => {
    e.preventDefault();
    if (!saveCustomName.trim()) {
      setSaveModelError("Please enter a model name.");
      return;
    }
    setSaveModelLoading(true);
    setSaveModelError(null);
    try {
      const response = await fetch("/api/saved-models", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          run_id: runId,
          model_name: showSaveModalFor.model_name,
          name: saveCustomName.trim()
        })
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to save model.");
      }
      
      // Reload saved models
      const savedRes = await fetch("/api/saved-models", {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (savedRes.ok) {
        const saved = await savedRes.json();
        setSavedModels(saved);
      }
      setShowSaveModalFor(null);
      setSaveCustomName("");
    } catch (err) {
      setSaveModelError(err.message);
    } finally {
      setSaveModelLoading(false);
    }
  };

  const openTestPredictor = async (model) => {
    setActiveTestingModel(model);
    setTestingModelLoading(true);
    setPredictionError(null);
    setPredictionResult(null);
    setPredictionInputs({});
    try {
      const response = await fetch(`/api/saved-models/${model.id}/details`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) throw new Error("Failed to load model features schema.");
      const data = await response.json();
      setTestingModelDetails(data);
      
      // Initialize inputs with default values
      const initialInputs = {};
      Object.entries(data.feature_profiles).forEach(([col, profile]) => {
        if (profile.type === "numeric") {
          initialInputs[col] = profile.mean !== undefined ? String(profile.mean.toFixed(2)) : "";
        } else {
          initialInputs[col] = profile.top_categories && profile.top_categories.length > 0 
            ? profile.top_categories[0].category 
            : "";
        }
      });
      setPredictionInputs(initialInputs);
    } catch (err) {
      setPredictionError(err.message);
    } finally {
      setTestingModelLoading(false);
    }
  };

  const runModelPrediction = async (e) => {
    e.preventDefault();
    setPredictionLoading(true);
    setPredictionError(null);
    setPredictionResult(null);
    try {
      const response = await fetch(`/api/saved-models/${activeTestingModel.id}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ inputs: predictionInputs })
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Prediction failed.");
      }
      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      setPredictionError(err.message);
    } finally {
      setPredictionLoading(false);
    }
  };

  const downloadSavedModel = async (model) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/runs/${model.run_id}/models/${model.model_name}/download`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) throw new Error("Failed to download model file.");
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `mlforge_${model.model_name.replace(" ", "_").lower()}_${model.run_id}.pkl`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadSavedModelExportCode = async (model) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/runs/${model.run_id}/models/${model.model_name}/export-code`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) throw new Error("Failed to fetch export configurations.");
      const data = await response.json();
      setRunId(model.run_id);
      setBestModelName(model.model_name);
      setExportCode(data.code);
      setExportInstructions(data.instructions);
      setStep(4); // Go to Deploy screen
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const deleteSavedModelClient = async (modelId) => {
    if (!window.confirm("Are you sure you want to delete this saved model registration?")) {
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`/api/saved-models/${modelId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) throw new Error("Failed to delete saved model.");
      setSavedModels(savedModels.filter(m => m.id !== modelId));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderDashboard = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      <div className="flex justify-between align-center">
        <div>
          <h2 style={{ fontSize: "1.8rem", marginBottom: "0.2rem" }} className="gradient-text">Welcome to your Workspace</h2>
          <p style={{ color: "var(--text-secondary)" }}>Manage your uploaded datasets and trained AutoML pipelines.</p>
        </div>
        <button className="btn btn-primary" onClick={() => setStep(1)} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <UploadCloud size={18} /> New Dataset
        </button>
      </div>

      {dashboardLoading ? (
        <div style={{ padding: "3rem", textAlign: "center" }}>
          <RefreshCw size={24} className="pulse-animation" style={{ color: "var(--color-primary)", marginBottom: "1rem" }} />
          <p>Loading workspace data...</p>
        </div>
      ) : (
        <div className="dashboard-widgets-grid">
          <div className="card">
            <h3 style={{ fontSize: "1.2rem", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <FileText size={18} style={{ color: "var(--color-secondary)" }} /> Your Datasets
            </h3>
            {userDatasets.length === 0 ? (
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>No datasets uploaded yet.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
                {userDatasets.map((ds) => (
                  <div
                    key={ds.id}
                    onClick={() => loadDatasetIntoWorkspace(ds.id, 1)}
                    style={{
                    padding: "1rem",
                    border: "1px solid var(--border-glass)",
                    borderRadius: "var(--radius-sm)",
                    background: "rgba(255, 255, 255, 0.02)",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    cursor: "pointer",
                  }}>
                    <div>
                      <strong style={{ display: "block", marginBottom: "0.2rem" }}>{ds.name}</strong>
                      <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                        Rows: {ds.total_rows} | Cols: {ds.total_cols}
                      </span>
                    </div>
                    <ChevronRight size={18} style={{ color: "var(--text-secondary)" }} />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card">
            <h3 style={{ fontSize: "1.2rem", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Cpu size={18} style={{ color: "var(--color-primary)" }} /> Your AutoML Runs
            </h3>
            {userRuns.length === 0 ? (
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>No models trained yet.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem", maxHeight: "350px", overflowY: "auto" }}>
                {userRuns.map((run) => (
                  <div key={run.id} style={{
                    padding: "1rem",
                    border: "1px solid var(--border-glass)",
                    borderRadius: "var(--radius-sm)",
                    background: "rgba(255, 255, 255, 0.02)",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    cursor: "pointer"
                  }}
                  onClick={() => {
                    setRunId(run.id);
                    setRunStatus(run.status);
                    setLeaderboard(run.leaderboard || []);
                    setBestModelName(run.best_model || "");
                    setComparedModelNames([]);
                    setRightPanelTab("insights");
                    setSelectedModelIndex(0);
                    setStep(3);
                    if (run.status !== "completed" && run.status !== "failed") {
                      pollTrainingStatus(run.id);
                    }
                  }}>
                    <div>
                      <strong style={{ display: "block", marginBottom: "0.2rem" }}>
                        {run.problem_type === "clustering" ? "Task: Clustering (Unsupervised)" : `Target: ${run.target}`}
                      </strong>
                      <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
                        Status: <span style={{ color: run.status === "completed" ? "var(--color-success)" : run.status === "failed" ? "var(--color-danger)" : "var(--color-warning)" }}>{run.status}</span>
                      </span>
                    </div>
                    <ChevronRight size={18} style={{ color: "var(--text-secondary)" }} />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Saved & Serving Models Section */}
          <div className="card" style={{ gridColumn: "span 2", marginTop: "1rem" }}>
            <h3 style={{ fontSize: "1.2rem", marginBottom: "1.2rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Sparkles size={18} style={{ color: "var(--color-warning)" }} /> Saved & Serving Models
            </h3>
            {savedModels.length === 0 ? (
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", textAlign: "center", padding: "2rem" }}>
                No saved models yet. Run AutoML and click "Save Model" on the leaderboard page to register them here.
              </p>
            ) : (
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Model Name</th>
                      <th>Type</th>
                      <th>Task</th>
                      <th>Target</th>
                      <th>Primary Metric</th>
                      <th style={{ textAlign: "right" }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {savedModels.map((m) => {
                      const metricName = m.problem_type === "classification" 
                        ? "F1-Score" 
                        : m.problem_type === "regression" 
                        ? "R² Score" 
                        : "Silhouette";
                      const metricVal = m.problem_type === "classification" 
                        ? m.metrics.f1 
                        : m.problem_type === "regression" 
                        ? m.metrics.r2 
                        : m.metrics.silhouette;
                      return (
                        <tr key={m.id}>
                          <td><strong>{m.name}</strong></td>
                          <td>{m.model_name}</td>
                          <td>
                            <span className={`col-badge badge-${m.problem_type === 'classification' ? 'categorical' : m.problem_type === 'regression' ? 'numeric' : 'unsupervised'}`} style={{ textTransform: "capitalize" }}>
                              {m.problem_type}
                            </span>
                          </td>
                          <td>{m.target}</td>
                          <td>
                            <span style={{ fontWeight: 600 }}>{metricName}: {metricVal.toFixed(4)}</span>
                          </td>
                          <td style={{ textAlign: "right" }}>
                            <div style={{ display: "inline-flex", gap: "0.4rem" }}>
                              <button
                                className="btn btn-primary"
                                style={{ padding: "0.3rem 0.6rem", fontSize: "0.75rem", borderRadius: "4px" }}
                                onClick={() => openTestPredictor(m)}
                              >
                                <Play size={12} /> Test Model
                              </button>
                              <button
                                className="btn btn-secondary"
                                style={{ padding: "0.3rem 0.6rem", fontSize: "0.75rem", borderRadius: "4px" }}
                                onClick={() => downloadSavedModel(m)}
                              >
                                <Download size={12} /> Download
                              </button>
                              <button
                                className="btn btn-secondary"
                                style={{ padding: "0.3rem 0.6rem", fontSize: "0.75rem", borderRadius: "4px" }}
                                onClick={() => loadSavedModelExportCode(m)}
                              >
                                <Code size={12} /> Export Code
                              </button>
                              <button
                                className="btn btn-secondary"
                                style={{ padding: "0.3rem 0.6rem", fontSize: "0.75rem", borderRadius: "4px", borderColor: "rgba(239, 68, 68, 0.4)", color: "var(--color-danger)" }}
                                onClick={() => deleteSavedModelClient(m.id)}
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  // Render Functions for different workflow screens
  const renderUploadStep = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      <div className="card text-center">
        <h2 style={{ fontSize: "1.8rem", marginBottom: "0.5rem" }}>Upload Your Dataset</h2>
        <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
          Upload a CSV file and MLForge will automatically profile columns and identify target variables.
        </p>

        <div
          className="upload-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <UploadCloud size={48} style={{ color: "var(--color-primary)", marginBottom: "1rem" }} />
          <p style={{ fontWeight: 600, fontSize: "1.1rem" }}>
            Drag & drop CSV file here, or <span style={{ color: "var(--color-primary)" }}>browse</span>
          </p>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.4rem" }}>
            Max dataset size: 10MB (.csv)
          </p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleFileUpload(e.target.files[0])}
            style={{ display: "none" }}
            accept=".csv"
          />
        </div>
      </div>

      {datasetId && (
        <div className="card">
          <div className="flex align-center justify-between mb-1">
            <div>
              <h3 style={{ fontSize: "1.4rem" }}>Dataset Profile: {datasetName}</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                Rows: <strong>{totalRows}</strong> | Columns: <strong>{totalCols}</strong>
              </p>
            </div>
            <button className="btn btn-primary" onClick={() => setStep(2)}>
              Configure AutoML <ChevronRight size={18} />
            </button>
          </div>

          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem", borderBottom: "1px solid var(--border-glass)" }}>
            <button
              className={`btn ${activeProfileTab === "preview" ? "btn-secondary" : ""}`}
              style={{ padding: "0.5rem 1rem", borderBottom: activeProfileTab === "preview" ? "2px solid var(--color-primary)" : "none", borderRadius: 0 }}
              onClick={() => setActiveProfileTab("preview")}
            >
              Data Preview
            </button>
            <button
              className={`btn ${activeProfileTab === "profile" ? "btn-secondary" : ""}`}
              style={{ padding: "0.5rem 1rem", borderBottom: activeProfileTab === "profile" ? "2px solid var(--color-primary)" : "none", borderRadius: 0 }}
              onClick={() => setActiveProfileTab("profile")}
            >
              Column Statistics
            </button>
          </div>

          {activeProfileTab === "preview" ? (
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    {Object.keys(sampleData[0] || {}).map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sampleData.slice(0, 8).map((row, idx) => (
                    <tr key={idx}>
                      {Object.values(row).map((val, i) => (
                        <td key={i}>{String(val)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="column-grid">
              {Object.entries(columnProfile).map(([colName, info]) => (
                <div className="card column-card" key={colName}>
                  <div className="column-header">
                    <span style={{ fontWeight: 600, fontFamily: "var(--font-heading)" }}>{colName}</span>
                    <span className={`col-badge badge-${info.type}`}>{info.type}</span>
                  </div>
                  <div>
                    <div className="col-stat">
                      <span>Missing Values</span>
                      <span>{info.missing_count} ({info.missing_pct}%)</span>
                    </div>
                    <div className="col-stat">
                      <span>Unique Values</span>
                      <span>{info.unique_count}</span>
                    </div>
                    {info.type === "numeric" && (
                      <>
                        <div className="col-stat">
                          <span>Mean | Median</span>
                          <span>{info.mean?.toFixed(2)} | {info.median?.toFixed(2)}</span>
                        </div>
                        <div className="col-stat">
                          <span>Range [Min - Max]</span>
                          <span>{info.min?.toFixed(1)} - {info.max?.toFixed(1)}</span>
                        </div>
                      </>
                    )}
                    {info.type === "categorical" && info.top_categories && (
                      <div style={{ marginTop: "0.5rem" }}>
                        <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 500 }}>Top Categories:</span>
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem", marginTop: "0.2rem" }}>
                          {info.top_categories.slice(0, 3).map((item, i) => (
                            <div key={i} className="col-stat" style={{ border: "none", padding: 0 }}>
                              <span style={{ color: "var(--text-primary)" }}>{item.category}</span>
                              <span style={{ color: "var(--text-muted)" }}>{item.count} counts</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderConfigStep = () => (
    <div className="card">
      <h2 style={{ fontSize: "1.8rem", marginBottom: "1.5rem" }} className="gradient-text">AutoML Training Settings</h2>

      <div className="config-grid">
        {/* Column 1: Settings */}
        <div>
          {problemType !== "clustering" ? (
            <div className="form-group">
              <label>Target Column (Variable to Predict)</label>
              <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                <option value="">-- Select Target --</option>
                {Object.keys(columnProfile).map((col) => (
                  <option key={col} value={col}>
                    {col} ({columnProfile[col].type})
                  </option>
                ))}
              </select>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                Auto-detected target column is selected by default.
              </span>
            </div>
          ) : (
            <div className="form-group">
              <label>Target Column</label>
              <select disabled value="">
                <option value="">None (Unsupervised Clustering)</option>
              </select>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                Unsupervised tasks do not require a target column.
              </span>
            </div>
          )}

          <div className="form-group">
            <label>ML Task Type</label>
            <select value={problemType} onChange={(e) => setProblemType(e.target.value)}>
              <option value="classification">Classification (Discrete Classes)</option>
              <option value="regression">Regression (Continuous values)</option>
              <option value="clustering">Clustering (Unsupervised)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Numerical Scaling</label>
            <select value={scaling} onChange={(e) => setScaling(e.target.value)}>
              <option value="standard">Standard Scaler (Subtract Mean & Divide Variance)</option>
              <option value="minmax">MinMax Scaler (Scale to 0-1 range)</option>
              <option value="robust">Robust Scaler (Handles Outliers using IQR)</option>
              <option value="none">No Scaling (Keep original distribution)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Numeric Imputation</label>
            <select value={imputation} onChange={(e) => setImputation(e.target.value)}>
              <option value="median">Median Imputation</option>
              <option value="mean">Mean Imputation</option>
              <option value="most_frequent">Mode Imputation</option>
            </select>
          </div>

          <div className="form-group">
            <label>Categorical Imputation</label>
            <select value={categoricalImputation} onChange={(e) => setCategoricalImputation(e.target.value)}>
              <option value="most_frequent">Most Frequent (Mode)</option>
              <option value="constant">Constant (&quot;missing&quot;)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Categorical Encoding</label>
            <select value={categoricalEncoding} onChange={(e) => setCategoricalEncoding(e.target.value)}>
              <option value="onehot">One-Hot Encoding</option>
              <option value="ordinal">Ordinal Encoding</option>
            </select>
          </div>

          <div className="form-group" style={{ marginTop: "1rem", borderTop: "1px solid var(--border-glass)", paddingTop: "1rem" }}>
            <label style={{ fontWeight: 600, display: "block", marginBottom: "0.5rem" }}>Select Models to Train</label>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {(problemType === "classification" 
                ? [
                    "Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Neural Network",
                    "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors", 
                    "Gradient Boosting", "AdaBoost", "Extra Trees", "Naive Bayes"
                  ]
                : problemType === "regression"
                ? [
                    "Linear Regression", "Random Forest", "XGBoost", "LightGBM", "Neural Network",
                    "Support Vector Regressor", "Decision Tree", "K-Nearest Neighbors",
                    "Gradient Boosting", "AdaBoost", "Extra Trees", "Ridge Regression", "Lasso Regression"
                  ]
                : [
                    "K-Means", "Birch", "Mean Shift", "Affinity Propagation", "Gaussian Mixture", "Mini Batch K-Means"
                  ]
              ).map((model) => (
                <label key={model} style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer", fontWeight: "normal", fontSize: "0.85rem" }}>
                  <input
                    type="checkbox"
                    checked={selectedModelsToTrain.includes(model)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedModelsToTrain([...selectedModelsToTrain, model]);
                      } else {
                        setSelectedModelsToTrain(selectedModelsToTrain.filter((m) => m !== model));
                      }
                    }}
                  />
                  <span>{model}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Column 2: Feature Selection */}
        <div>
          <label style={{ display: "block", marginBottom: "0.8rem", fontWeight: 600 }}>Select Feature Columns</label>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", marginBottom: "1rem" }}>
            Click a feature to preview preprocessing. Uncheck columns you do not wish to train on.
          </p>
          <div style={{
            maxHeight: "420px",
            overflowY: "auto",
            border: "1px solid var(--border-glass)",
            padding: "0.5rem",
            borderRadius: "var(--radius-sm)",
            background: "rgba(0, 0, 0, 0.15)"
          }}>
            {Object.keys(columnProfile).map((col) => {
              const isTarget = col === targetColumn;
              const isConstant = columnProfile[col].type === "constant";
              const isIdOrConstant = columnProfile[col].type === "id" || isConstant;
              const isActive = activePreviewFeature === col;
              return (
                <div
                  key={col}
                  onClick={() => {
                    if (!isTarget && !isConstant) {
                      setActivePreviewFeature(col);
                    }
                  }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.8rem",
                    padding: "0.5rem 0.6rem",
                    borderRadius: "var(--radius-sm)",
                    cursor: (isTarget || isConstant) ? "not-allowed" : "pointer",
                    background: isActive ? "rgba(99, 102, 241, 0.15)" : "transparent",
                    border: isActive ? "1px solid var(--color-primary)" : "1px solid transparent",
                    marginBottom: "0.2rem",
                  }}
                >
                  <input
                    type="checkbox"
                    id={`feat-${col}`}
                    disabled={isTarget || isIdOrConstant}
                    checked={selectedFeatures.includes(col) && !isTarget}
                    onChange={(e) => {
                      e.stopPropagation();
                      if (e.target.checked) {
                        setSelectedFeatures([...selectedFeatures, col]);
                      } else {
                        setSelectedFeatures(selectedFeatures.filter((f) => f !== col));
                      }
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <label htmlFor={`feat-${col}`} style={{
                    color: isTarget ? "var(--text-muted)" : "var(--text-primary)",
                    textDecoration: isTarget ? "line-through" : "none",
                    cursor: (isTarget || isIdOrConstant) ? "not-allowed" : "pointer",
                    fontSize: "0.85rem",
                    flex: 1,
                  }}>
                    {col} <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>({columnProfile[col].type})</span>
                    {isTarget && " (Target)"}
                    {isIdOrConstant && " (Dropped)"}
                  </label>
                  {isActive && <Eye size={14} style={{ color: "var(--color-primary)", flexShrink: 0 }} />}
                </div>
              );
            })}
          </div>
        </div>

        {/* Column 3: Live Preview */}
        <div>
          <label style={{ marginBottom: "0.8rem", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Eye size={16} style={{ color: "var(--color-secondary)" }} />
            Live Preprocessing Preview
          </label>
          {!activePreviewFeature ? (
            <div style={{
              padding: "2rem 1rem",
              textAlign: "center",
              border: "1px dashed var(--border-glass)",
              borderRadius: "var(--radius-sm)",
              color: "var(--text-secondary)",
              fontSize: "0.85rem",
            }}>
              Select a feature column to see before vs. after preprocessing.
            </div>
          ) : previewLoading ? (
            <div style={{ padding: "2rem", textAlign: "center" }}>
              <RefreshCw size={24} className="pulse-animation" style={{ color: "var(--color-primary)", marginBottom: "0.5rem" }} />
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Computing preview for <strong>{activePreviewFeature}</strong>...</p>
            </div>
          ) : previewError ? (
            <div style={{
              padding: "1rem",
              border: "1px solid var(--color-danger)",
              borderRadius: "var(--radius-sm)",
              color: "var(--color-danger)",
              fontSize: "0.85rem",
            }}>
              {previewError}
            </div>
          ) : previewData ? (
            <div>
              {previewData.preview_available === false ? (
                <div style={{
                  padding: "1.5rem 1rem",
                  border: "1px dashed var(--border-glass)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text-secondary)",
                  fontSize: "0.85rem",
                }}>
                  {previewData.message}
                </div>
              ) : (
                <>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.8rem" }}>
                <strong>{previewData.feature}</strong> ({previewData.column_type})
                {" · "}
                {previewData.column_type === "numeric" ? `${scaling} scaling + ${imputation} imputation` : `${categoricalEncoding} encoding + ${categoricalImputation} imputation`}
              </p>
              <div className="table-container" style={{ maxHeight: "380px", overflowY: "auto", overflowX: "auto" }}>
                <table>
                  <thead>
                    <tr>
                      <th style={{ minWidth: "60px" }}>#</th>
                      <th>Before</th>
                      {previewData.after_columns.map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.before.map((val, idx) => (
                      <tr key={idx}>
                        <td style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>{idx + 1}</td>
                        <td style={{ fontWeight: 500 }}>{val === null || val === undefined ? <span style={{ color: "var(--color-warning)" }}>null</span> : String(val)}</td>
                        {previewData.after_columns.map((col) => (
                          <td key={col} style={{ color: "var(--color-secondary)" }}>
                            {previewData.after[idx]?.[col] ?? "—"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
                </>
              )}
            </div>
          ) : null}
        </div>
      </div>

      <div style={{ display: "flex", gap: "1rem", marginTop: "2rem", justifyContent: "flex-end" }}>
        <button className="btn btn-secondary" onClick={() => setStep(1)}>
          Back
        </button>
        <button className="btn btn-primary" onClick={startTraining}>
          Start Parallel Training <Play size={18} />
        </button>
      </div>
    </div>
  );

  const renderLeaderboardStep = () => {
    const selectedModel = leaderboard[selectedModelIndex];

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
        {/* Training Wait Screen */}
        {(runStatus === "pending" || runStatus === "training") && (
          <div className="card text-center pulse-animation" style={{ padding: "4rem 2rem" }}>
            <Cpu size={56} style={{ color: "var(--color-primary)", marginBottom: "1.5rem" }} className="pulse-animation" />
            <h3 style={{ fontSize: "1.6rem", marginBottom: "0.8rem" }}>Training ML models in parallel...</h3>
            <p style={{ color: "var(--text-secondary)", maxWidth: "600px", margin: "0 auto 1.5rem" }}>
              Fitting selected classification, regression, or clustering estimators to your feature inputs.
              This will evaluate them side-by-side using optimized cross-validation.
            </p>
            <div style={{ display: "flex", justifyContent: "center", gap: "1.5rem", marginTop: "2rem" }}>
              {["Linear Models", "Forests/Trees", "XGBoost", "LightGBM", "Neural Net", "Clustering"].map((name, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <div style={{
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    background: runStatus === "training" ? "var(--color-primary)" : "var(--border-glass)",
                    boxShadow: runStatus === "training" ? "0 0 8px var(--color-primary)" : "none"
                  }} />
                  <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>{name}</span>
                </div>
              ))}
            </div>
            <button className="btn btn-secondary" style={{ marginTop: "2.5rem" }} onClick={navigateToDashboard}>
              Back to Dashboard
            </button>
          </div>
        )}

        {runStatus === "failed" && (
          <div className="card text-center" style={{ borderLeft: "4px solid var(--color-danger)" }}>
            <AlertTriangle size={48} className="text-danger" style={{ marginBottom: "1rem" }} />
            <h3 style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>Model Training Failed</h3>
            <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem" }}>{runError}</p>
            <div style={{ display: "flex", justifyContent: "center", gap: "1rem" }}>
              <button className="btn btn-secondary" onClick={() => setStep(2)}>
                Adjust Settings
              </button>
              <button className="btn btn-secondary" onClick={navigateToDashboard}>
                Back to Dashboard
              </button>
            </div>
          </div>
        )}

        {/* Dashboard Completed Screen */}
        {runStatus === "completed" && (
          <div className="dashboard-grid">
            {/* Leaderboard Table Column */}
            <div className="card" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div className="flex justify-between align-center">
                <h3 style={{ fontSize: "1.3rem" }}>Training Leaderboard</h3>
                <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  Ranked by {problemType === "classification" ? "F1-Score" : problemType === "regression" ? "R² Score" : "Silhouette Coefficient"}
                </span>
              </div>

              <div style={{ border: "1px solid var(--border-glass)", borderRadius: "var(--radius-sm)", overflowX: "auto" }}>
                <div style={{ minWidth: "650px" }}>
                  {/* Header row */}
                  <div className="leaderboard-row" style={{ display: "grid", gridTemplateColumns: "30px 45px 1.4fr 0.9fr 0.9fr 130px", gap: "0.5rem", alignItems: "center", background: "var(--bg-card)", fontWeight: 600, borderBottom: "1px solid var(--border-glass)", padding: "1rem 1.2rem" }}>
                    <span></span>
                    <span>Rank</span>
                    <span>Model</span>
                    <span>{problemType === "classification" ? "F1" : problemType === "regression" ? "R²" : "Silhouette"}</span>
                    <span>{problemType === "classification" ? "Accuracy" : problemType === "regression" ? "RMSE" : "Davies-Bouldin"}</span>
                    <span>Actions</span>
                  </div>

                  {/* Rows */}
                  {leaderboard.map((model, idx) => {
                    const score = problemType === "classification" 
                      ? model.metrics.f1 
                      : problemType === "regression" 
                      ? model.metrics.r2 
                      : model.metrics.silhouette;
                    const secScore = problemType === "classification" 
                      ? model.metrics.accuracy 
                      : problemType === "regression" 
                      ? model.metrics.rmse 
                      : model.metrics.davies_bouldin;
                    const isSelected = selectedModelIndex === idx;
                    const isSaved = savedModels.some(
                      (sm) => sm.run_id === runId && sm.model_name === model.model_name
                    );

                    return (
                      <div
                        key={model.model_name}
                        className={`leaderboard-row ${model.is_best ? "best" : ""}`}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "30px 45px 1.4fr 0.9fr 0.9fr 130px",
                          gap: "0.5rem",
                          alignItems: "center",
                          padding: "1rem 1.2rem",
                          background: isSelected ? "rgba(255, 255, 255, 0.04)" : "",
                          borderLeft: isSelected ? "3px solid var(--color-primary)" : model.is_best ? "3px solid var(--color-success)" : ""
                        }}
                        onClick={() => setSelectedModelIndex(idx)}
                      >
                        <input
                          type="checkbox"
                          checked={comparedModelNames.includes(model.model_name)}
                          onChange={(e) => {
                            e.stopPropagation();
                            if (e.target.checked) {
                              setComparedModelNames([...comparedModelNames, model.model_name]);
                            } else {
                              setComparedModelNames(comparedModelNames.filter((n) => n !== model.model_name));
                            }
                          }}
                          onClick={(e) => e.stopPropagation()}
                        />
                        <span style={{ fontWeight: 600 }}>#{model.rank}</span>
                        <span style={{ fontWeight: 500, display: "flex", alignItems: "center", gap: "0.4rem", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {model.model_name}
                          {model.is_best && (
                            <span style={{
                              fontSize: "0.6rem",
                              background: "var(--color-success)",
                              color: "#fff",
                              padding: "0.1rem 0.4rem",
                              borderRadius: "20px"
                            }}>BEST</span>
                          )}
                        </span>
                        <span>{(score !== undefined && score !== null) ? score.toFixed(4) : "N/A"}</span>
                        <span>{(secScore !== undefined && secScore !== null) ? secScore.toFixed(4) : "N/A"}</span>
                        <div style={{ display: "flex", gap: "0.4rem" }}>
                          {isSaved ? (
                            <span className="text-success" style={{ fontSize: "0.75rem", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.2rem" }}>
                              <CheckCircle size={12} /> Saved
                            </span>
                          ) : (
                            <button
                              className="btn btn-primary"
                              style={{ padding: "0.3rem 0.5rem", fontSize: "0.7rem", borderRadius: "4px" }}
                              onClick={(e) => {
                                e.stopPropagation();
                                setShowSaveModalFor(model);
                                setSaveCustomName(`${model.model_name} (${problemType})`);
                                setSaveModelError(null);
                              }}
                            >
                              Save
                            </button>
                          )}
                          <button
                            className="btn btn-secondary"
                            style={{ padding: "0.3rem 0.5rem", fontSize: "0.7rem", borderRadius: "4px" }}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedModelIndex(idx);
                              loadExportCode();
                            }}
                          >
                            Deploy
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
                <button className="btn btn-secondary" onClick={() => setStep(2)}>
                  Configure Again
                </button>
                <button className="btn btn-secondary" onClick={navigateToDashboard} style={{ flex: 1 }}>
                  Back to Dashboard
                </button>
                <button className="btn btn-primary" onClick={loadExportCode}>
                  Deploy Best Model <ChevronRight size={18} />
                </button>
              </div>
            </div>

            {/* Model Insights & Comparison Column */}
            {selectedModel && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                <div style={{ display: "flex", gap: "0.5rem", borderBottom: "1px solid var(--border-glass)", paddingBottom: "0.5rem" }}>
                  <button
                    className={`btn ${rightPanelTab === "insights" ? "btn-primary" : "btn-secondary"}`}
                    style={{ padding: "0.4rem 1rem", fontSize: "0.8rem", borderRadius: "var(--radius-sm)" }}
                    onClick={() => setRightPanelTab("insights")}
                  >
                    Model Insights ({selectedModel.model_name})
                  </button>
                  <button
                    className={`btn ${rightPanelTab === "comparison" ? "btn-primary" : "btn-secondary"}`}
                    style={{ padding: "0.4rem 1rem", fontSize: "0.8rem", borderRadius: "var(--radius-sm)", display: "flex", alignItems: "center", gap: "0.3rem" }}
                    onClick={() => setRightPanelTab("comparison")}
                  >
                    Compare Selected ({comparedModelNames.length})
                  </button>
                </div>

                {rightPanelTab === "insights" ? (
                  <div className="card">
                    <h3 style={{ fontSize: "1.4rem", marginBottom: "1rem" }} className="gradient-text">
                      {selectedModel.model_name} Insights
                    </h3>

                    {/* Performance stats cards */}
                    <div className="stats-grid">
                      {problemType === "classification" ? (
                        <>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>Accuracy</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{(selectedModel.metrics.accuracy * 100).toFixed(1)}%</h4>
                          </div>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>F1-Score</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.f1.toFixed(3)}</h4>
                          </div>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>Recall</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.recall.toFixed(3)}</h4>
                          </div>
                        </>
                      ) : problemType === "regression" ? (
                        <>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>R² Score</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.r2.toFixed(3)}</h4>
                          </div>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>MAE</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.mae.toFixed(2)}</h4>
                          </div>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>RMSE</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.rmse.toFixed(2)}</h4>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>Silhouette</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.silhouette.toFixed(3)}</h4>
                          </div>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>Davies-Bouldin</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.davies_bouldin.toFixed(3)}</h4>
                          </div>
                          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)", padding: "1rem", textAlign: "center" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>Calinski-Harabasz</span>
                            <h4 style={{ fontSize: "1.6rem", marginTop: "0.3rem" }}>{selectedModel.metrics.calinski_harabasz.toFixed(1)}</h4>
                          </div>
                        </>
                      )}
                    </div>

                    {/* Confusion Matrix (for Classification Only) */}
                    {problemType === "classification" && selectedModel.metrics.confusion_matrix && (
                      <div style={{ marginBottom: "1.5rem" }}>
                        <h4 style={{ fontSize: "1rem", marginBottom: "0.6rem" }}>Confusion Matrix</h4>
                        <div style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          border: "1px solid var(--border-glass)",
                          padding: "1rem",
                          borderRadius: "var(--radius-sm)",
                          background: "rgba(0, 0, 0, 0.1)"
                        }}>
                          <div style={{ display: "flex", gap: "4px" }}>
                            {/* Y-axis label */}
                            <div style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", textAlign: "center", fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                              Actual Label
                            </div>
                            
                            <div>
                              {/* Matrix columns headers */}
                              <div style={{ display: "flex", marginLeft: "60px", marginBottom: "4px", gap: "4px" }}>
                                {selectedModel.metrics.confusion_matrix.classes.map((cls) => (
                                  <div key={cls} style={{ width: "60px", textAlign: "center", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                                    Pred {cls}
                                  </div>
                                ))}
                              </div>

                              {/* Matrix Rows */}
                              {selectedModel.metrics.confusion_matrix.matrix.map((row, rowIdx) => (
                                <div key={rowIdx} style={{ display: "flex", alignItems: "center", gap: "4px", marginBottom: "4px" }}>
                                  <div style={{ width: "60px", textAlign: "right", paddingRight: "8px", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                                    Act {selectedModel.metrics.confusion_matrix.classes[rowIdx]}
                                  </div>
                                  {row.map((val, colIdx) => {
                                    // Compute intensity color based on correctness
                                    const isCorrect = rowIdx === colIdx;
                                    const rowSum = row.reduce((a, b) => a + b, 0);
                                    const ratio = rowSum > 0 ? val / rowSum : 0;
                                    const opacity = 0.1 + ratio * 0.8;
                                    const cellBg = isCorrect
                                      ? `rgba(16, 185, 129, ${opacity})`
                                      : `rgba(239, 68, 68, ${opacity})`;

                                    return (
                                      <div
                                        key={colIdx}
                                        style={{
                                          width: "60px",
                                          height: "40px",
                                          display: "flex",
                                          alignItems: "center",
                                          justifyContent: "center",
                                          background: cellBg,
                                          borderRadius: "4px",
                                          fontSize: "0.85rem",
                                          fontWeight: 600,
                                          color: "#fff",
                                          boxShadow: isCorrect && ratio > 0.5 ? "0 0 5px rgba(16, 185, 129, 0.3)" : ""
                                        }}
                                      >
                                        {val}
                                      </div>
                                    );
                                  })}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Feature Importance View */}
                    {selectedModel.explainability && (
                      <div>
                        <h4 style={{ fontSize: "1.05rem", marginBottom: "0.8rem", color: "var(--text-primary)" }}>
                          Global Feature Importances
                        </h4>
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                          {selectedModel.explainability.global_importance.slice(0, 5).map((featInfo) => (
                            <div key={featInfo.feature} style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                              <div className="flex justify-between" style={{ fontSize: "0.8rem" }}>
                                <span>{featInfo.feature}</span>
                                <span style={{ fontWeight: 600, color: "var(--color-secondary)" }}>
                                  {(featInfo.importance * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div style={{ height: "8px", background: "var(--border-glass)", borderRadius: "4px", overflow: "hidden" }}>
                                <div style={{
                                  width: `${featInfo.importance * 100}%`,
                                  height: "100%",
                                  background: "linear-gradient(90deg, var(--color-primary) 0%, var(--color-secondary) 100%)",
                                  borderRadius: "4px",
                                  boxShadow: "0 0 6px var(--color-primary-glow)"
                                }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="card">
                    <h3 style={{ fontSize: "1.4rem", marginBottom: "1rem" }} className="gradient-text">
                      Model Comparison
                    </h3>
                    {comparedModelNames.length < 2 ? (
                      <div style={{ padding: "3rem 1rem", textAlign: "center", color: "var(--text-secondary)" }}>
                        <Cpu size={32} style={{ marginBottom: "1rem", color: "var(--color-primary)" }} />
                        <h4 style={{ color: "var(--text-primary)" }}>Select Models to Compare</h4>
                        <p style={{ fontSize: "0.85rem", marginTop: "0.5rem" }}>
                          Check the boxes next to 2 or more models in the leaderboard table to view side-by-side metrics.
                        </p>
                      </div>
                    ) : (
                      <div>
                        <div className="table-container" style={{ margin: "1rem 0" }}>
                          <table>
                            <thead>
                              <tr>
                                <th>Metric</th>
                                {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => (
                                  <th key={m.model_name}>{m.model_name}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {problemType === "classification" ? (
                                <>
                                  <tr>
                                    <td><strong>F1-Score</strong></td>
                                    {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => {
                                      const val = m.metrics.f1;
                                      const maxVal = Math.max(...leaderboard.filter(x => comparedModelNames.includes(x.model_name)).map(x => x.metrics.f1));
                                      const isMax = val === maxVal;
                                      return (
                                        <td key={m.model_name} style={{ color: isMax ? "var(--color-success)" : "", fontWeight: isMax ? 600 : 400 }}>
                                          {val.toFixed(4)} {isMax && "🏆"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                  <tr>
                                    <td><strong>Accuracy</strong></td>
                                    {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => {
                                      const val = m.metrics.accuracy;
                                      const maxVal = Math.max(...leaderboard.filter(x => comparedModelNames.includes(x.model_name)).map(x => x.metrics.accuracy));
                                      const isMax = val === maxVal;
                                      return (
                                        <td key={m.model_name} style={{ color: isMax ? "var(--color-success)" : "", fontWeight: isMax ? 600 : 400 }}>
                                          {val.toFixed(4)} {isMax && "🏆"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                  <tr>
                                    <td><strong>Precision</strong></td>
                                    {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => {
                                      const val = m.metrics.precision;
                                      const maxVal = Math.max(...leaderboard.filter(x => comparedModelNames.includes(x.model_name)).map(x => x.metrics.precision));
                                      const isMax = val === maxVal;
                                      return (
                                        <td key={m.model_name} style={{ color: isMax ? "var(--color-success)" : "", fontWeight: isMax ? 600 : 400 }}>
                                          {val.toFixed(4)} {isMax && "🏆"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                  <tr>
                                    <td><strong>Recall</strong></td>
                                    {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => {
                                      const val = m.metrics.recall;
                                      const maxVal = Math.max(...leaderboard.filter(x => comparedModelNames.includes(x.model_name)).map(x => x.metrics.recall));
                                      const isMax = val === maxVal;
                                      return (
                                        <td key={m.model_name} style={{ color: isMax ? "var(--color-success)" : "", fontWeight: isMax ? 600 : 400 }}>
                                          {val.toFixed(4)} {isMax && "🏆"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                </>
                              ) : (
                                <>
                                  <tr>
                                    <td><strong>R² Score</strong></td>
                                    {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => {
                                      const val = m.metrics.r2;
                                      const maxVal = Math.max(...leaderboard.filter(x => comparedModelNames.includes(x.model_name)).map(x => x.metrics.r2));
                                      const isMax = val === maxVal;
                                      return (
                                        <td key={m.model_name} style={{ color: isMax ? "var(--color-success)" : "", fontWeight: isMax ? 600 : 400 }}>
                                          {val.toFixed(4)} {isMax && "🏆"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                  <tr>
                                    <td><strong>MAE</strong></td>
                                    {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => {
                                      const val = m.metrics.mae;
                                      const minVal = Math.min(...leaderboard.filter(x => comparedModelNames.includes(x.model_name)).map(x => x.metrics.mae));
                                      const isMin = val === minVal;
                                      return (
                                        <td key={m.model_name} style={{ color: isMin ? "var(--color-success)" : "", fontWeight: isMin ? 600 : 400 }}>
                                          {val.toFixed(4)} {isMin && "🏆"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                  <tr>
                                    <td><strong>RMSE</strong></td>
                                    {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map(m => {
                                      const val = m.metrics.rmse;
                                      const minVal = Math.min(...leaderboard.filter(x => comparedModelNames.includes(x.model_name)).map(x => x.metrics.rmse));
                                      const isMin = val === minVal;
                                      return (
                                        <td key={m.model_name} style={{ color: isMin ? "var(--color-success)" : "", fontWeight: isMin ? 600 : 400 }}>
                                          {val.toFixed(4)} {isMin && "🏆"}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                </>
                              )}
                            </tbody>
                          </table>
                        </div>

                        <div style={{ marginTop: "1.5rem" }}>
                          <h4 style={{ fontSize: "1rem", marginBottom: "0.8rem", color: "var(--text-primary)" }}>
                            Visual Metric Comparison: {problemType === "classification" ? "F1-Score" : "R² Score"}
                          </h4>
                          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                            {leaderboard.filter(m => comparedModelNames.includes(m.model_name)).map((m) => {
                              const val = problemType === "classification" ? m.metrics.f1 : m.metrics.r2;
                              const pct = Math.max(0, Math.min(100, val * 100));
                              const isBestRun = m.model_name === leaderboard[0].model_name;
                              return (
                                <div key={m.model_name} style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                                    <span>{m.model_name}</span>
                                    <strong style={{ color: isBestRun ? "var(--color-success)" : "var(--color-secondary)" }}>{val.toFixed(4)}</strong>
                                  </div>
                                  <div style={{ height: "10px", background: "rgba(255,255,255,0.05)", borderRadius: "5px", overflow: "hidden" }}>
                                    <div style={{
                                      width: `${pct}%`,
                                      height: "100%",
                                      background: isBestRun 
                                        ? "linear-gradient(90deg, var(--color-success) 0%, #34d399 100%)" 
                                        : "linear-gradient(90deg, var(--color-primary) 0%, var(--color-secondary) 100%)",
                                      borderRadius: "5px",
                                      boxShadow: isBestRun ? "0 0 8px rgba(16, 185, 129, 0.4)" : "0 0 6px var(--color-primary-glow)"
                                    }} />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* SHAP Explanation Force Plot Card (Insights Tab Only) */}
                {rightPanelTab === "insights" && selectedModel.explainability && selectedModel.explainability.shap_detail && (
                  <div className="card" style={{ marginTop: "1.5rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.8rem" }}>
                      <Sparkles size={18} style={{ color: "var(--color-warning)" }} />
                      <h3 style={{ fontSize: "1.1rem" }}>Model SHAP Explanations</h3>
                    </div>
                    <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", marginBottom: "1.2rem" }}>
                      Visualizing target push-forces for a test set prediction. Features dragging **left (red)** push the
                      prediction lower, features dragging **right (green)** push it higher.
                    </p>

                    <div style={{
                      display: "flex",
                      flexDirection: "column",
                      border: "1px solid var(--border-glass)",
                      padding: "1rem",
                      borderRadius: "var(--radius-sm)",
                      background: "rgba(0,0,0,0.15)"
                    }}>
                      <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", display: "flex", justifyContent: "space-between", marginBottom: "0.8rem" }}>
                        <span>Base Output Value: <strong>{selectedModel.explainability.shap_detail.base_value.toFixed(2)}</strong></span>
                        <span style={{ color: "var(--color-primary)", fontWeight: 600 }}>SHAP Force Plot</span>
                      </div>

                      {/* Displaying SHAP values for top features of Sample Row #1 */}
                      <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
                        {selectedModel.explainability.global_importance.slice(0, 4).map((fInfo, idx) => {
                          const shapValues = selectedModel.explainability.shap_detail.shap_values;
                          const featIdx = selectedModel.explainability.feature_names.indexOf(fInfo.feature);
                          const valSum = shapValues.reduce((sum, valRow) => sum + (valRow[featIdx] || 0), 0);
                          const meanVal = valSum / shapValues.length;

                          const isPositive = meanVal >= 0;
                          const widthPct = Math.min(Math.abs(meanVal) * 100 * 3, 50); // Scale up for visual representation
                          
                          return (
                            <div key={fInfo.feature} style={{ display: "flex", alignItems: "center", fontSize: "0.8rem" }}>
                              <div style={{ width: "120px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                                {fInfo.feature}
                              </div>
                              <div style={{ flex: 1, display: "flex", alignItems: "center", position: "relative", height: "16px" }}>
                                <div style={{ position: "absolute", left: "50%", width: "1px", height: "16px", background: "var(--border-glass)" }} />
                                
                                {isPositive ? (
                                  <div style={{
                                    position: "absolute",
                                    left: "50%",
                                    height: "10px",
                                    width: `${widthPct}%`,
                                    background: "var(--color-success)",
                                    borderRadius: "0 2px 2px 0",
                                    boxShadow: "0 0 5px rgba(16, 185, 129, 0.3)"
                                  }} />
                                ) : (
                                  <div style={{
                                    position: "absolute",
                                    right: "50%",
                                    height: "10px",
                                    width: `${widthPct}%`,
                                    background: "var(--color-danger)",
                                    borderRadius: "2px 0 0 2px",
                                    boxShadow: "0 0 5px rgba(239, 68, 68, 0.3)"
                                  }} />
                                )}
                              </div>
                              <div style={{ width: "60px", textAlign: "right", color: isPositive ? "var(--color-success)" : "var(--color-danger)", fontWeight: 600 }}>
                                {isPositive ? "+" : ""}{meanVal.toFixed(3)}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderExportStep = () => (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div className="flex align-center justify-between">
        <div>
          <h2 style={{ fontSize: "1.8rem" }} className="gradient-text">Deploy Best Pipeline: {bestModelName}</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.2rem" }}>
            Download the serialized binary file and run this local FastAPI server wrapper.
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button
            className="btn btn-secondary"
            onClick={() => {
              setStep(3);
            }}
          >
            Back to Leaderboard
          </button>
          <button
            className="btn btn-secondary"
            onClick={navigateToDashboard}
          >
            Back to Dashboard
          </button>
        </div>
      </div>

      <div className="export-grid">
        {/* Left column - Downloads */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)" }}>
            <h3 style={{ fontSize: "1.1rem", marginBottom: "0.8rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Download size={18} style={{ color: "var(--color-primary)" }} /> Download Pipeline
            </h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", marginBottom: "1.2rem" }}>
              The download is a serialized Scikit-Learn binary `.pkl` that contains the complete preprocessing sequence
              and the trained model estimators.
            </p>
            <button
              onClick={handleDownloadModel}
              className="btn btn-primary"
              style={{ width: "100%", display: "flex", gap: "0.5rem", justifyContent: "center" }}
            >
              <Download size={16} /> Download Pipeline (.pkl)
            </button>
          </div>

          <div className="card" style={{ background: "rgba(255, 255, 255, 0.01)" }}>
            <h3 style={{ fontSize: "1.1rem", marginBottom: "0.8rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Info size={18} style={{ color: "var(--color-secondary)" }} /> Pipeline Structure
            </h3>
            <ul style={{ fontSize: "0.8rem", color: "var(--text-secondary)", paddingLeft: "1.2rem", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
              <li><strong>Removes Columns</strong>: ID indexes, constant columns.</li>
              <li><strong>Numerical Imputation</strong>: {imputation} replacement.</li>
              <li><strong>Categorical Imputation</strong>: {categoricalImputation === "constant" ? "Constant (\"missing\")" : "Most frequent (mode)"}.</li>
              <li><strong>Categorical Encoding</strong>: {categoricalEncoding === "ordinal" ? "OrdinalEncoder" : "OneHotEncoder (expanding labels)"}.</li>
              <li><strong>Feature Scaling</strong>: {scaling === "none" ? "Disabled" : scaling + " scaling"}.</li>
              <li><strong>Model Estimator</strong>: {bestModelName} pipeline.</li>
            </ul>
          </div>
        </div>

        {/* Right column - Code Export */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div>
            <div className="flex justify-between align-center">
              <span style={{ fontSize: "0.9rem", fontWeight: 600 }}>FastAPI Deployment Server (`main.py`)</span>
              <button
                className="btn btn-secondary"
                style={{ padding: "0.3rem 0.6rem", fontSize: "0.75rem", display: "flex", gap: "0.3rem" }}
                onClick={() => copyToClipboard(exportCode, "code")}
              >
                {copiedCode ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                {copiedCode ? "Copied" : "Copy Code"}
              </button>
            </div>
            <pre className="code-container">
              <code>{exportCode}</code>
            </pre>
          </div>

          <div>
            <div className="flex justify-between align-center" style={{ marginTop: "1rem" }}>
              <span style={{ fontSize: "0.9rem", fontWeight: 600 }}>Command Line Setup Instructions</span>
              <button
                className="btn btn-secondary"
                style={{ padding: "0.3rem 0.6rem", fontSize: "0.75rem", display: "flex", gap: "0.3rem" }}
                onClick={() => copyToClipboard(exportInstructions, "instructions")}
              >
                {copiedInstructions ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                {copiedInstructions ? "Copied" : "Copy"}
              </button>
            </div>
            <div style={{
              background: "rgba(0, 0, 0, 0.2)",
              padding: "1rem",
              borderRadius: "var(--radius-sm)",
              fontSize: "0.8rem",
              color: "var(--text-secondary)",
              whiteSpace: "pre-line",
              border: "1px solid var(--border-glass)",
              marginTop: "0.5rem"
            }}>
              {exportInstructions}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAuthScreen = () => {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "65vh", padding: "2rem 0" }}>
        <div className="card" style={{ width: "100%", maxWidth: "420px", display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div style={{ textAlign: "center" }}>
            <Cpu size={40} style={{ color: "var(--color-primary)", marginBottom: "1rem" }} className="pulse-animation" />
            <h2 style={{ fontSize: "1.8rem", marginBottom: "0.4rem" }} className="gradient-text">
              {authMode === "login" ? "Welcome to MLForge" : "Create Account"}
            </h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
              {authMode === "login" ? "Sign in to access your AutoML workspace" : "Register a new account to get started"}
            </p>
          </div>

          {authError && (
            <div className="card" style={{ borderLeft: "4px solid var(--color-danger)", padding: "0.8rem 1rem", display: "flex", gap: "0.6rem", alignItems: "center", background: "rgba(239, 68, 68, 0.05)" }}>
              <AlertTriangle size={18} className="text-danger" style={{ flexShrink: 0 }} />
              <span style={{ color: "var(--color-danger)", fontSize: "0.8rem", fontWeight: 500 }}>{authError}</span>
            </div>
          )}

          <form onSubmit={handleAuthSubmit} style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label htmlFor="auth-username" style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <User size={14} style={{ color: "var(--text-secondary)" }} /> Username
              </label>
              <input
                id="auth-username"
                type="text"
                value={authUsername}
                onChange={(e) => setAuthUsername(e.target.value)}
                placeholder="Enter username"
                disabled={authLoading}
                style={{ width: "100%" }}
              />
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label htmlFor="auth-password" style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <Lock size={14} style={{ color: "var(--text-secondary)" }} /> Password
              </label>
              <input
                id="auth-password"
                type="password"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                placeholder="Enter password"
                disabled={authLoading}
                style={{ width: "100%" }}
              />
            </div>

            {authMode === "signup" && (
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label htmlFor="auth-confirm-password" style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                  <Lock size={14} style={{ color: "var(--text-secondary)" }} /> Confirm Password
                </label>
                <input
                  id="auth-confirm-password"
                  type="password"
                  value={authConfirmPassword}
                  onChange={(e) => setAuthConfirmPassword(e.target.value)}
                  placeholder="Confirm your password"
                  disabled={authLoading}
                  style={{ width: "100%" }}
                />
              </div>
            )}

            <button type="submit" className="btn btn-primary" style={{ width: "100%", marginTop: "0.5rem", display: "flex", gap: "0.5rem", justifyContent: "center" }} disabled={authLoading}>
              {authLoading ? (
                <>
                  <RefreshCw size={16} className="pulse-animation" /> 
                  {authMode === "login" ? "Logging in..." : "Signing up..."}
                </>
              ) : (
                <>
                  <LogIn size={16} />
                  {authMode === "login" ? "Log In" : "Sign Up"}
                </>
              )}
            </button>
          </form>

          <div style={{ textAlign: "center", fontSize: "0.85rem", borderTop: "1px solid var(--border-glass)", paddingTop: "1rem" }}>
            <span style={{ color: "var(--text-secondary)" }}>
              {authMode === "login" ? "Don't have an account? " : "Already have an account? "}
            </span>
            <button
              onClick={() => {
                setAuthMode(authMode === "login" ? "signup" : "login");
                setAuthError(null);
                setAuthUsername("");
                setAuthPassword("");
                setAuthConfirmPassword("");
              }}
              style={{
                background: "none",
                border: "none",
                color: "var(--color-primary)",
                cursor: "pointer",
                fontWeight: 600,
                padding: 0,
                fontFamily: "inherit"
              }}
            >
              {authMode === "login" ? "Sign Up" : "Log In"}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div>
      {/* Header navbar */}
      <nav className="navbar">
        <a href="#" className="nav-logo">
          <Cpu size={24} style={{ color: "var(--color-primary)" }} />
          <span>ML<span className="gradient-text">Forge</span></span>
        </a>
        <div className="nav-links" style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
          {user && (
            <span style={{ fontSize: "0.9rem", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <User size={16} style={{ color: "var(--color-primary)" }} />
              <strong>{user.username}</strong>
            </span>
          )}
          <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--color-success)" }} />
            Connected to Local Mongo & Python Server
          </span>
          {user && (
            <button 
              onClick={handleLogout}
              className="btn btn-secondary" 
              style={{ padding: "0.4rem 0.8rem", fontSize: "0.8rem", display: "flex", alignItems: "center", gap: "0.3rem" }}
            >
              <LogOut size={14} /> Log Out
            </button>
          )}
        </div>
      </nav>

      {/* Page Container */}
      <div className="container">
        {!token || !user ? (
          renderAuthScreen()
        ) : (
          <>
            {/* Progress Stepper bar */}
            <div className="stepper">
              <div className={`step ${step >= 1 ? "active" : ""} ${step > 1 ? "completed" : ""}`} onClick={() => datasetId && setStep(1)}>
                <div className="step-circle">{step > 1 ? <CheckCircle size={16} /> : "1"}</div>
                <div className="step-label">Upload CSV</div>
              </div>
              <div className={`step ${step >= 2 ? "active" : ""} ${step > 2 ? "completed" : ""}`} onClick={() => datasetId && setStep(2)}>
                <div className="step-circle">{step > 2 ? <CheckCircle size={16} /> : "2"}</div>
                <div className="step-label">Configure</div>
              </div>
              <div className={`step ${step >= 3 ? "active" : ""} ${step > 3 ? "completed" : ""}`} onClick={() => runId && setStep(3)}>
                <div className="step-circle">{step > 3 ? <CheckCircle size={16} /> : "3"}</div>
                <div className="step-label">Leaderboard</div>
              </div>
              <div className={`step ${step >= 4 ? "active" : ""} ${step > 4 ? "completed" : ""}`} onClick={() => runStatus === "completed" && setStep(4)}>
                <div className="step-circle">4</div>
                <div className="step-label">Export</div>
              </div>
            </div>

            {/* Global Error Banner */}
            {error && (
              <div className="card" style={{ borderLeft: "4px solid var(--color-danger)", marginBottom: "1.5rem", padding: "1rem", display: "flex", gap: "0.8rem", alignItems: "center" }}>
                <AlertTriangle size={20} className="text-danger" />
                <div>
                  <span style={{ fontWeight: 600, color: "var(--color-danger)" }}>Error: </span>
                  <span style={{ color: "var(--text-primary)", fontSize: "0.85rem" }}>{error}</span>
                </div>
              </div>
            )}

            {/* Loading Spinner */}
            {loading && (
              <div style={{
                position: "fixed",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: "rgba(0,0,0,0.5)",
                zIndex: 1000,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backdropFilter: "blur(4px)"
              }}>
                <div className="card text-center" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.8rem", padding: "2rem" }}>
                  <RefreshCw size={36} className="pulse-animation" style={{ color: "var(--color-primary)" }} />
                  <span style={{ fontSize: "0.9rem", fontWeight: 500 }}>Processing request...</span>
                </div>
              </div>
            )}

            {/* Core Steps Switcher */}
            {step === 0 && renderDashboard()}
            {step === 1 && renderUploadStep()}
            {step === 2 && renderConfigStep()}
            {step === 3 && renderLeaderboardStep()}
            {step === 4 && renderExportStep()}

            {/* Save Model Modal */}
            {showSaveModalFor && (
              <div style={{
                position: "fixed",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: "rgba(0,0,0,0.6)",
                backdropFilter: "blur(8px)",
                zIndex: 1100,
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}>
                <div className="card" style={{ width: "100%", maxWidth: "450px", display: "flex", flexDirection: "column", gap: "1.2rem" }}>
                  <h3 style={{ fontSize: "1.3rem" }} className="gradient-text">Register Model</h3>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                    Save **{showSaveModalFor.model_name}** to your workspace dashboard for serving and exporting.
                  </p>

                  {saveModelError && (
                    <div style={{ color: "var(--color-danger)", fontSize: "0.8rem", background: "rgba(239, 68, 68, 0.05)", padding: "0.5rem 0.8rem", borderRadius: "var(--radius-sm)", borderLeft: "3px solid var(--color-danger)" }}>
                      {saveModelError}
                    </div>
                  )}

                  <form onSubmit={confirmSaveModel} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label htmlFor="save-name">Custom Model Name</label>
                      <input
                        id="save-name"
                        type="text"
                        value={saveCustomName}
                        onChange={(e) => setSaveCustomName(e.target.value)}
                        placeholder="e.g. Titanic RandomForest Classifier"
                        disabled={saveModelLoading}
                        required
                        style={{ width: "100%" }}
                      />
                    </div>

                    <div style={{ display: "flex", gap: "0.8rem", justifyContent: "flex-end", marginTop: "0.5rem" }}>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ padding: "0.5rem 1rem", fontSize: "0.85rem" }}
                        onClick={() => {
                          setShowSaveModalFor(null);
                          setSaveCustomName("");
                        }}
                        disabled={saveModelLoading}
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="btn btn-primary"
                        style={{ padding: "0.5rem 1rem", fontSize: "0.85rem" }}
                        disabled={saveModelLoading}
                      >
                        {saveModelLoading ? <RefreshCw size={14} className="pulse-animation" /> : "Save Model"}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}

            {/* Interactive Predictor Modal */}
            {activeTestingModel && (
              <div style={{
                position: "fixed",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: "rgba(0,0,0,0.6)",
                backdropFilter: "blur(8px)",
                zIndex: 1100,
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}>
                <div className="card" style={{ width: "100%", maxWidth: "850px", maxHeight: "90vh", overflow: "hidden", display: "flex", flexDirection: "column", gap: "1rem" }}>
                  <div className="flex justify-between align-center" style={{ borderBottom: "1px solid var(--border-glass)", paddingBottom: "0.8rem" }}>
                    <div>
                      <h3 style={{ fontSize: "1.3rem" }} className="gradient-text">Interactive Predictor</h3>
                      <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", marginTop: "0.2rem" }}>
                        Testing model: <strong>{activeTestingModel.name}</strong> ({activeTestingModel.model_name})
                      </p>
                    </div>
                    <button
                      className="btn btn-secondary"
                      style={{ padding: "0.3rem 0.6rem", fontSize: "0.8rem", borderRadius: "4px" }}
                      onClick={() => {
                        setActiveTestingModel(null);
                        setTestingModelDetails(null);
                        setPredictionResult(null);
                      }}
                    >
                      Close
                    </button>
                  </div>

                  {testingModelLoading ? (
                    <div style={{ padding: "4rem", textAlign: "center" }}>
                      <RefreshCw size={28} className="pulse-animation" style={{ color: "var(--color-primary)", marginBottom: "1rem" }} />
                      <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>Loading model details and schema...</p>
                    </div>
                  ) : testingModelDetails ? (
                    <div className="test-model-grid">
                      
                      {/* Features Input Form */}
                      <form onSubmit={runModelPrediction} style={{ display: "flex", flexDirection: "column", gap: "1rem", overflowY: "auto", paddingRight: "0.5rem", maxHeight: "60vh" }}>
                        <h4 style={{ fontSize: "1rem", borderBottom: "1px solid var(--border-glass)", paddingBottom: "0.3rem" }}>Input Features</h4>
                        {Object.entries(testingModelDetails.feature_profiles).map(([col, profile]) => (
                          <div key={col} className="form-group" style={{ marginBottom: "0.5rem" }}>
                            <label style={{ display: "flex", justifyContent: "space-between" }}>
                              <span>{col}</span>
                              <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)" }}>({profile.type})</span>
                            </label>
                            {profile.type === "numeric" ? (
                              <div>
                                <input
                                  type="number"
                                  step="any"
                                  value={predictionInputs[col] ?? ""}
                                  onChange={(e) => setPredictionInputs({ ...predictionInputs, [col]: e.target.value })}
                                  placeholder={profile.mean !== undefined ? `Mean: ${profile.mean.toFixed(2)}` : "Enter number"}
                                  style={{ width: "100%", padding: "0.6rem" }}
                                />
                                {profile.min !== undefined && profile.max !== undefined && (
                                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "block", marginTop: "0.2rem" }}>
                                    Valid range: {profile.min.toFixed(1)} - {profile.max.toFixed(1)}
                                  </span>
                                )}
                              </div>
                            ) : (
                              <select
                                value={predictionInputs[col] ?? ""}
                                onChange={(e) => setPredictionInputs({ ...predictionInputs, [col]: e.target.value })}
                                style={{ width: "100%", padding: "0.6rem" }}
                              >
                                <option value="">-- Select Option --</option>
                                {(profile.top_categories || []).map((cat) => (
                                  <option key={cat.category} value={cat.category}>
                                    {cat.category}
                                  </option>
                                ))}
                              </select>
                            )}
                          </div>
                        ))}
                        
                        <div style={{ marginTop: "1rem", position: "sticky", bottom: 0, background: "var(--bg-card)", paddingTop: "0.5rem", zIndex: 10 }}>
                          <button type="submit" className="btn btn-primary" style={{ width: "100%", display: "flex", gap: "0.5rem", justifyContent: "center" }} disabled={predictionLoading}>
                            {predictionLoading ? (
                              <>
                                <RefreshCw size={16} className="pulse-animation" /> Computing prediction...
                              </>
                            ) : (
                              <>
                                <Play size={16} /> Run Inference
                              </>
                            )}
                          </button>
                        </div>
                      </form>

                      {/* Prediction Outcomes Panel */}
                      <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem", background: "rgba(0,0,0,0.15)", padding: "1.2rem", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-glass)", overflowY: "auto", maxHeight: "60vh" }}>
                        <h4 style={{ fontSize: "1rem", borderBottom: "1px solid var(--border-glass)", paddingBottom: "0.3rem" }}>Prediction Outcome</h4>
                        
                        {predictionError && (
                          <div style={{ color: "var(--color-danger)", fontSize: "0.85rem", background: "rgba(239, 68, 68, 0.05)", padding: "0.8rem", borderRadius: "var(--radius-sm)", borderLeft: "4px solid var(--color-danger)" }}>
                            <strong>Inference failed:</strong>
                            <p style={{ marginTop: "0.2rem" }}>{predictionError}</p>
                          </div>
                        )}

                        {!predictionResult && !predictionError && (
                          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-secondary)", textAlign: "center", padding: "2rem 0" }}>
                            <Play size={24} style={{ color: "var(--color-primary)", marginBottom: "0.5rem" }} className="pulse-animation" />
                            <p style={{ fontSize: "0.85rem" }}>Enter input features and click **Run Inference** to compute predicted values.</p>
                          </div>
                        )}

                        {predictionResult && (
                          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                            {/* Prediction Result Badge */}
                            <div style={{
                              padding: "1.2rem",
                              borderRadius: "var(--radius-sm)",
                              background: "rgba(255, 255, 255, 0.02)",
                              border: "1px solid var(--border-glass)",
                              textAlign: "center"
                            }}>
                              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.5rem" }}>
                                {activeTestingModel.problem_type === "clustering" 
                                  ? "Predicted Cluster Assignment" 
                                  : `Model Output Prediction (${activeTestingModel.target})`}
                              </span>
                              <strong style={{ fontSize: "2rem", color: "var(--color-success)", textShadow: "0 0 10px rgba(16, 185, 129, 0.25)" }}>
                                {predictionResult.prediction}
                              </strong>
                            </div>

                            {/* Probabilities Output (Classification Only) */}
                            {predictionResult.probabilities && (
                              <div>
                                <h5 style={{ fontSize: "0.9rem", marginBottom: "0.6rem" }}>Class Confidence Scores</h5>
                                <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
                                  {Object.entries(predictionResult.probabilities).map(([cls, prob]) => {
                                    const pct = (prob * 100).toFixed(1);
                                    const isPredicted = String(predictionResult.prediction) === String(cls);
                                    return (
                                      <div key={cls} style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem" }}>
                                          <span style={{ fontWeight: isPredicted ? 600 : 400, color: isPredicted ? "var(--text-primary)" : "var(--text-secondary)" }}>
                                            Class {cls} {isPredicted && " (predicted)"}
                                          </span>
                                          <span style={{ fontWeight: 600, color: isPredicted ? "var(--color-success)" : "var(--text-secondary)" }}>{pct}%</span>
                                        </div>
                                        <div style={{ height: "8px", background: "rgba(255,255,255,0.04)", borderRadius: "4px", overflow: "hidden" }}>
                                          <div style={{
                                            width: `${pct}%`,
                                            height: "100%",
                                            background: isPredicted 
                                              ? "linear-gradient(90deg, var(--color-success) 0%, #34d399 100%)"
                                              : "rgba(255,255,255,0.15)",
                                            borderRadius: "4px"
                                          }} />
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                    </div>
                  ) : (
                    <div style={{ padding: "3rem", textAlign: "center", color: "var(--color-danger)" }}>
                      Failed to load details.
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
