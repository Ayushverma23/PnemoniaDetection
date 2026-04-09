"use client";

import { useState } from "react";
import Sidebar from "../../components/Sidebar";
import FileUpload from "../../components/FileUpload";
import { AlertCircle, CheckCircle2, ArrowRight } from "lucide-react";

export default function TuberculosisPage() {
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
            const response = await fetch("http://localhost:8000/api/v1/predict/tuberculosis", {
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
                    <h2 style={{ fontSize: "1.5rem", fontWeight: "bold" }}>Tuberculosis Analysis</h2>
                    <p style={{ color: "var(--text-muted)" }}>Upload a Chest X-Ray for automated Tuberculosis detection.</p>
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
                                {loading ? "Analyzing..." : <>Run TB Analysis <ArrowRight size={18} /></>}
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
                                Tuberculosis Report
                            </h3>

                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem", marginBottom: "2rem" }}>
                                <div>
                                    <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginBottom: "0.25rem" }}>Prediction</p>
                                    <div style={{
                                        fontSize: "1.5rem", fontWeight: "bold",
                                        color: result.prediction === "Normal" ? "var(--success)" : "var(--danger)",
                                        display: "flex", alignItems: "center", gap: "0.5rem"
                                    }}>
                                        {result.prediction === "Normal" ? (
                                            <><CheckCircle2 size={24} /> Negative (Normal)</>
                                        ) : (
                                            <><AlertCircle size={24} /> {result.prediction}</>
                                        )}
                                    </div>
                                </div>

                                <div>
                                    <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Highest Confidence</p>
                                    <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                                        <div style={{ flex: 1, backgroundColor: "#e2e8f0", height: "12px", borderRadius: "6px", overflow: "hidden" }}>
                                            <div style={{
                                                width: `${result.confidence * 100}%`,
                                                backgroundColor: result.prediction === "Normal" ? "var(--success)" : "var(--danger)",
                                                height: "100%", borderRadius: "6px", transition: "width 1s ease-out"
                                            }} />
                                        </div>
                                        <span style={{ fontWeight: "bold", minWidth: "3rem" }}>
                                            {(result.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Annotated Image */}
                            {result.annotated_image_base64 && (
                                <div style={{ marginTop: "1rem" }}>
                                    <p style={{ fontSize: "0.875rem", fontWeight: "bold", marginBottom: "0.5rem" }}>Detection Results</p>
                                    <div style={{ display: "flex", justifyContent: "center", backgroundColor: "#f1f5f9", padding: "1rem", borderRadius: "var(--radius)" }}>
                                        <img
                                            src={`data:image/png;base64,${result.annotated_image_base64}`}
                                            alt="TB Detection"
                                            style={{ maxWidth: "100%", maxHeight: "400px", borderRadius: "8px", border: "1px solid var(--border)" }}
                                        />
                                    </div>
                                </div>
                            )}
                        </section>
                    )}
                </div>
            </main>
        </>
    );
}
