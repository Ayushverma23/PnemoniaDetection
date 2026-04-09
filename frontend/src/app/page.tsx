"use client";

import { useState } from "react";
import Sidebar from "../components/Sidebar";
import FileUpload from "../components/FileUpload";
import { AlertCircle, CheckCircle2, ArrowRight } from "lucide-react";

export default function Home() {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://localhost:8000/api/v1/predict/pneumonia", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Prediction failed");
            }

            const data = await response.json();
            setResult(data);
        } catch (err: any) {
            console.error(err);
            setError(err.message || "Something went wrong. Please check if backend is running.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <Sidebar />
            <main style={{ flex: 1, padding: "2rem", overflowY: "auto" }}>
                <header style={{ marginBottom: "2rem" }}>
                    <h2 style={{ fontSize: "1.5rem", fontWeight: "bold" }}>Pneumonia Analysis</h2>
                    <p style={{ color: "var(--text-muted)" }}>Upload a Chest X-Ray (DICOM/Image) for automated Pneumonia detection.</p>
                </header>

                <div style={{ maxWidth: "800px", margin: "0 auto" }}>
                    {/* Upload Section */}
                    <section style={{ marginBottom: "2rem" }}>
                        <FileUpload onFileSelect={setFile} />

                        <div style={{ marginTop: "1.5rem", display: "flex", justifyContent: "flex-end" }}>
                            <button
                                className="btn btn-primary"
                                disabled={!file || loading}
                                onClick={handleUpload}
                                style={{ opacity: !file || loading ? 0.6 : 1, display: "flex", alignItems: "center", gap: "0.5rem" }}
                            >
                                {loading ? "Analyzing..." : <>Run Pneumonia Analysis <ArrowRight size={18} /></>}
                            </button>
                        </div>
                    </section>

                    {/* Error Message */}
                    {error && (
                        <div style={{
                            backgroundColor: "#fef2f2", color: "#991b1b", padding: "1rem",
                            borderRadius: "var(--radius)", border: "1px solid #fecaca",
                            display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "2rem"
                        }}>
                            <AlertCircle size={20} />
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Results Section */}
                    {result && (
                        <section className="card">
                            <h3 style={{ fontSize: "1.1rem", fontWeight: "bold", marginBottom: "1.5rem", borderBottom: "1px solid var(--border)", paddingBottom: "0.75rem" }}>
                                Analysis Report
                            </h3>

                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
                                <div>
                                    <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginBottom: "0.25rem" }}>Prediction</p>
                                    <div style={{
                                        fontSize: "1.5rem", fontWeight: "bold",
                                        color: result.prediction === "Pneumonia" ? "var(--danger)" : "var(--success)",
                                        display: "flex", alignItems: "center", gap: "0.5rem"
                                    }}>
                                        {result.prediction === "Pneumonia" ? (
                                            <>
                                                <AlertCircle size={24} /> Positive (Pneumonia)
                                            </>
                                        ) : (
                                            <>
                                                <CheckCircle2 size={24} /> Negative (Normal)
                                            </>
                                        )}
                                    </div>
                                </div>

                                <div>
                                    <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Model Confidence</p>
                                    <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                                        <div style={{ flex: 1, backgroundColor: "#e2e8f0", height: "12px", borderRadius: "6px", overflow: "hidden" }}>
                                            <div style={{
                                                width: `${result.probability * 100}%`,
                                                backgroundColor: result.prediction === "Pneumonia" ? "var(--danger)" : "var(--success)",
                                                height: "100%", borderRadius: "6px", transition: "width 1s ease-out"
                                            }} />
                                        </div>
                                        <span style={{ fontWeight: "bold", minWidth: "3rem" }}>
                                            {(result.probability * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
                                        Probability of Pneumonia classification.
                                    </p>
                                </div>
                            </div>

                            {/* Grad-CAM Heatmap Visualization */}
                            {result.heatmap_base64 && (
                                <div style={{ marginTop: "2rem" }}>
                                    <p style={{ fontSize: "0.875rem", fontWeight: "bold", marginBottom: "0.5rem" }}>Grad-CAM Visualization</p>
                                    <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
                                        Red areas indicate regions contributing most to the prediction.
                                    </p>
                                    <div style={{
                                        display: "flex", justifyContent: "center",
                                        backgroundColor: "#f1f5f9", padding: "1rem", borderRadius: "var(--radius)"
                                    }}>
                                        <img
                                            src={`data:image/png;base64,${result.heatmap_base64}`}
                                            alt="Grad-CAM Heatmap"
                                            style={{ maxWidth: "100%", maxHeight: "400px", borderRadius: "8px", border: "1px solid var(--border)" }}
                                        />
                                    </div>
                                </div>
                            )}

                            <div style={{ marginTop: "2rem", backgroundColor: "#f8fafc", padding: "1rem", borderRadius: "var(--radius)" }}>
                                <p style={{ fontSize: "0.875rem", fontWeight: "bold", marginBottom: "0.5rem" }}>File Details:</p>
                                <code style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>{result.filename}</code>
                            </div>
                        </section>
                    )}
                </div>
            </main>
        </>
    );
}
