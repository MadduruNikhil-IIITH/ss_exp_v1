document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("analyze-btn");
    const contextInput = document.getElementById("context-input");
    const modelSelect = document.getElementById("model-select");
    
    const guideText = document.getElementById("saliency-guide-text");
    const highlightViewer = document.getElementById("highlight-viewer");
    
    const qgCard = document.getElementById("qg-card");
    const selectedSentText = document.getElementById("selected-sent-text");
    
    const generateQBtn = document.getElementById("generate-q-btn");
    const generatedQOutput = document.getElementById("generated-q-output");
    const generatedQText = document.getElementById("generated-q-text");
    
    const quizBtn = document.getElementById("quiz-btn");
    const quizCard = document.getElementById("quiz-card");
    const quizItemsContainer = document.getElementById("quiz-items-container");
    
    const stepQa = document.getElementById("step-qa");
    const solveQBtn = document.getElementById("solve-q-btn");
    const solverOutput = document.getElementById("solver-output");
    const solverAnswerText = document.getElementById("solver-answer-text");
    const verificationStatus = document.getElementById("verification-status");
    const statusText = document.getElementById("status-text");
    
    const tooltip = document.getElementById("features-tooltip");
    const tooltipList = document.getElementById("tooltip-features-list");
    
    const priorityTableCard = document.getElementById("priority-table-card");
    const priorityTableBody = document.getElementById("priority-table-body");

    // Render Priority Table (Top-5 Sentences Ranked High -> Low)
    const renderPriorityTable = (sentences) => {
        if (!sentences || !sentences.length) return;
        const sorted = [...sentences].sort((a, b) => b.probability - a.probability);
        
        priorityTableBody.innerHTML = "";
        sorted.slice(0, 5).forEach((sent, idx) => {
            const tr = document.createElement("tr");
            tr.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
            
            const rankColors = ["#ffb703", "#00f5d4", "#7b2cbf", "#3a86ff", "#ff006e"];
            const color = rankColors[idx] || "#3a86ff";
            
            const probPct = (sent.probability * 100).toFixed(1);
            const llmProbPct = sent.llm_judge_probability !== undefined ? (sent.llm_judge_probability * 100).toFixed(1) : "35.0";
            const rstN = sent.features && sent.features["rst_nuclearity_ratio"] !== undefined ? parseFloat(sent.features["rst_nuclearity_ratio"]).toFixed(2) : "0.50";
            const surpDrop = sent.features && sent.features["surp_deletion_drop"] !== undefined ? (parseFloat(sent.features["surp_deletion_drop"]) >= 0 ? "+" : "") + parseFloat(sent.features["surp_deletion_drop"]).toFixed(2) : "+1.20";
            
            tr.innerHTML = `
                <td style="padding: 0.65rem 0.5rem; text-align: center;">
                    <span class="badge" style="background: ${color}22; color: ${color}; border: 1px solid ${color}; font-weight: 700;">#${idx + 1}</span>
                </td>
                <td style="padding: 0.65rem 0.5rem; text-align: center; font-weight: 600; color: var(--text-muted);">Sent #${sent.sentence_idx}</td>
                <td style="padding: 0.65rem 0.5rem; color: #fff; font-size: 0.85rem;">${sent.text}</td>
                <td style="padding: 0.65rem 0.5rem; text-align: center;">
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 2px;">
                        <strong style="color: ${color}; font-size: 0.9rem;">${probPct}%</strong>
                        <div style="width: 50px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
                            <div style="width: ${probPct}%; height: 100%; background: ${color};"></div>
                        </div>
                    </div>
                </td>
                <td style="padding: 0.65rem 0.5rem; text-align: center;">
                    <span class="badge" style="background: rgba(157,78,221,0.15); color: var(--accent-purple); border: 1px solid var(--accent-purple); font-size: 0.8rem;">${llmProbPct}%</span>
                </td>
                <td style="padding: 0.65rem 0.5rem; text-align: center; color: var(--text-muted); font-size: 0.8rem;">${rstN}</td>
                <td style="padding: 0.65rem 0.5rem; text-align: center; color: var(--text-muted); font-size: 0.8rem;">${surpDrop}</td>
                <td style="padding: 0.65rem 0.5rem; text-align: center;">
                    <button class="btn secondary-btn select-table-btn" style="padding: 0.25rem 0.5rem; font-size: 0.75rem; border-radius: 4px;">
                        <i class="fa-solid fa-crosshairs"></i> Select
                    </button>
                </td>
            `;
            
            tr.querySelector(".select-table-btn").addEventListener("click", () => {
                selectSentence(sent.sentence_idx);
            });
            
            priorityTableBody.appendChild(tr);
        });
        
        priorityTableCard.classList.remove("hidden");
    };

    // Default Sample Text
    contextInput.value = "Chrysler ended production of their full-sized luxury sedans at the end of the 1981 model year, moving instead to a full front-wheel drive lineup for 1982 (except for the M-body Dodge Diplomat/Plymouth Gran Fury and Chrysler New Yorker Fifth Avenue sedans). Production of the Imperial ended in 1983. In 1985, Chrysler showed the Chrysler LeBaron GTS. It was a five-door hatchback built on a modified version of the K-car platform.";

    // Tooltip Mouse tracking
    const showTooltip = (event, features) => {
        tooltipList.innerHTML = "";
        for (const [key, val] of Object.entries(features)) {
            const li = document.createElement("li");
            li.innerHTML = `<strong>${key}</strong>: ${val}`;
            tooltipList.appendChild(li);
        }
        tooltip.classList.remove("hidden");
        moveTooltip(event);
    };

    const moveTooltip = (event) => {
        tooltip.style.left = (event.pageX + 15) + "px";
        tooltip.style.top = (event.pageY + 15) + "px";
    };

    const hideTooltip = () => {
        tooltip.classList.add("hidden");
    };

    // Click selector for sentence
    const selectSentence = (idx) => {
        selectedIdx = idx;
        const spans = document.querySelectorAll(".salient-span");
        spans.forEach((span, i) => {
            if (i === idx) {
                span.classList.add("selected");
            } else {
                span.classList.remove("selected");
            }
        });
        
        // Populate QG card
        selectedSentText.innerText = activeSentences[idx].text;
        qgCard.classList.remove("hidden");
        
        // Reset steps
        generatedQOutput.classList.add("hidden");
        stepQa.classList.add("hidden");
        solverOutput.classList.add("hidden");
    };

    // Analyze Saliency Click
    analyzeBtn.addEventListener("click", async () => {
        const context = contextInput.value.trim();
        const modelName = modelSelect.value;
        
        if (!context) {
            alert("Please input a context paragraph.");
            return;
        }
        
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector("span").innerText = "Analyzing Context...";
        analyzeBtn.querySelector("i").className = "fa-solid fa-spinner fa-spin";
        
        try {
            const res = await fetch("/predict_saliency", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ context, model_name: modelName })
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Server error.");
            }
            
            const data = await res.json();
            activeSentences = data.sentences;
            
            // Render Highlighted HTML
            highlightViewer.innerHTML = "";
            activeSentences.forEach((sent, idx) => {
                const span = document.createElement("span");
                span.className = "salient-span";
                span.innerText = sent.text + " ";
                
                // Color scaling based on probability (gold background glow)
                const opacity = 0.08 + sent.probability * 0.4;
                span.style.backgroundColor = `rgba(255, 183, 3, ${opacity})`;
                span.style.borderBottom = `2px dashed rgba(255, 183, 3, ${sent.probability * 0.8})`;
                
                // Hover Features Tooltip
                span.addEventListener("mouseenter", (e) => showTooltip(e, sent.features));
                span.addEventListener("mousemove", moveTooltip);
                span.addEventListener("mouseleave", hideTooltip);
                
                // Selection Trigger
                span.addEventListener("click", () => selectSentence(idx));
                
                highlightViewer.appendChild(span);
            });
            
            guideText.classList.add("hidden");
            highlightViewer.classList.remove("hidden");
            
            // Render Top-5 Priority Table
            renderPriorityTable(activeSentences);
            
            // Auto-select the top predicted salient sentence
            selectSentence(data.selected_idx);
            
        } catch (e) {
            alert("Error running analysis: " + e.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector("span").innerText = "Analyze Sentence Saliency";
            analyzeBtn.querySelector("i").className = "fa-solid fa-wand-magic-sparkles";
        }
    });

    // Generate Question Click
    generateQBtn.addEventListener("click", async () => {
        if (selectedIdx === null) return;
        const context = contextInput.value.trim();
        const sentence = activeSentences[selectedIdx].text;
        
        generateQBtn.disabled = true;
        generateQBtn.querySelector("span").innerText = "Generating Question...";
        generateQBtn.querySelector("i").className = "fa-solid fa-spinner fa-spin";
        
        try {
            const res = await fetch("/generate_question", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ context, sentence })
            });
            
            if (!res.ok) throw new Error("QG Agent failure.");
            
            const data = await res.json();
            generatedQText.innerText = data.question;
            generatedQOutput.classList.remove("hidden");
            stepQa.classList.remove("hidden");
            solverOutput.classList.add("hidden");
            
        } catch (e) {
            alert("Error: " + e.message);
        } finally {
            generateQBtn.disabled = false;
            generateQBtn.querySelector("span").innerText = "Generate Question";
            generateQBtn.querySelector("i").className = "fa-solid fa-gears";
        }
    });

    // Solve & Verify Click
    solveQBtn.addEventListener("click", async () => {
        if (selectedIdx === null) return;
        const context = contextInput.value.trim();
        const question = generatedQText.innerText;
        const targetSentence = activeSentences[selectedIdx].text;
        
        solveQBtn.disabled = true;
        solveQBtn.querySelector("span").innerText = "Solving question...";
        solveQBtn.querySelector("i").className = "fa-solid fa-spinner fa-spin";
        
        try {
            const res = await fetch("/answer_question", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ context, question, target_sentence: targetSentence })
            });
            
            if (!res.ok) throw new Error("QA Solver failure.");
            
            const data = await res.json();
            solverAnswerText.innerText = data.predicted_answer;
            
            // Render indicator
            if (data.verified) {
                verificationStatus.className = "status-indicator verified";
                verificationStatus.querySelector("i").className = "fa-solid fa-circle-check";
                statusText.innerText = "Verified Acceptable";
            } else {
                verificationStatus.className = "status-indicator rejected";
                verificationStatus.querySelector("i").className = "fa-solid fa-circle-xmark";
                statusText.innerText = "Incorrect/Unspecific";
            }
            
            solverOutput.classList.remove("hidden");
            
        } catch (e) {
            alert("Error: " + e.message);
        } finally {
            solveQBtn.disabled = false;
            solveQBtn.querySelector("span").innerText = "Verify with QA Solver Agent";
            solveQBtn.querySelector("i").className = "fa-solid fa-user-robot";
        }
    });

    // Generate Quiz Click
    quizBtn.addEventListener("click", async () => {
        const context = contextInput.value.trim();
        const modelName = modelSelect.value;
        
        if (!context) {
            alert("Please input a context paragraph.");
            return;
        }
        
        quizBtn.disabled = true;
        quizBtn.querySelector("span").innerText = "Generating Quiz...";
        quizBtn.querySelector("i").className = "fa-solid fa-spinner fa-spin";
        
        try {
            // First, trigger saliency highlighting so user can see it
            const saliencyRes = await fetch("/predict_saliency", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ context, model_name: modelName })
            });
            
            if (!saliencyRes.ok) throw new Error("Saliency prediction failure.");
            
            const saliencyData = await saliencyRes.json();
            activeSentences = saliencyData.sentences;
            
            // Render Highlighted HTML
            highlightViewer.innerHTML = "";
            activeSentences.forEach((sent, idx) => {
                const span = document.createElement("span");
                span.className = "salient-span";
                span.innerText = sent.text + " ";
                
                const opacity = 0.08 + sent.probability * 0.4;
                span.style.backgroundColor = `rgba(255, 183, 3, ${opacity})`;
                span.style.borderBottom = `2px dashed rgba(255, 183, 3, ${sent.probability * 0.8})`;
                
                span.addEventListener("mouseenter", (e) => showTooltip(e, sent.features));
                span.addEventListener("mousemove", moveTooltip);
                span.addEventListener("mouseleave", hideTooltip);
                span.addEventListener("click", () => selectSentence(idx));
                
                highlightViewer.appendChild(span);
            });
            
            guideText.classList.add("hidden");
            highlightViewer.classList.remove("hidden");
            
            // Hide single mode QG card
            qgCard.classList.add("hidden");
            
            // Call generate quiz endpoint (Fast batched QG & QA Agent verification)
            const quizRes = await fetch("/generate_quiz", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ context, model_name: modelName })
            });
            
            if (!quizRes.ok) throw new Error("Quiz generation failure.");
            
            const quizData = await quizRes.json();
            
            // Clear and render Quiz Items
            quizItemsContainer.innerHTML = "";
            
            quizData.quiz_items.forEach((item, index) => {
                const itemDiv = document.createElement("div");
                itemDiv.className = "glass-inner";
                itemDiv.style.borderRadius = "8px";
                itemDiv.style.padding = "1rem";
                itemDiv.style.display = "flex";
                itemDiv.style.flexDirection = "column";
                itemDiv.style.gap = "0.75rem";
                
                const verifiedClass = item.verified ? "verified" : "rejected";
                const verifiedIcon = item.verified ? "fa-circle-check" : "fa-circle-xmark";
                const verifiedText = item.verified ? "Verified Acceptable" : "Unverified/Incorrect";
                
                itemDiv.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge" style="color: var(--accent-blue);">Question ${index + 1}</span>
                        <div class="status-indicator ${verifiedClass}" style="height: 24px; padding: 0 0.5rem; font-size: 0.75rem;">
                            <i class="fa-solid ${verifiedIcon}"></i>
                            <span>${verifiedText}</span>
                        </div>
                    </div>
                    <p style="font-size: 1.05rem; font-weight: 500; font-style: italic;">"${item.question}"</p>
                    
                    <div class="input-group">
                        <input type="text" placeholder="Type your answer here..." style="background: rgba(0,0,0,0.3); border: 1px solid var(--panel-border); border-radius: 4px; padding: 0.5rem 0.75rem; color: #fff; outline: none; font-size: 0.9rem; width: 100%;">
                    </div>
                    
                    <div style="display: flex; gap: 0.75rem;">
                        <button class="btn secondary-btn show-ans-btn" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">
                            <i class="fa-solid fa-eye"></i> Show Correct Answer
                        </button>
                    </div>
                    
                    <div class="ans-reveal hidden" style="background: rgba(0,0,0,0.2); border-left: 3px solid var(--yellow-salient); padding: 0.50rem 0.75rem; font-size: 0.85rem; color: var(--text-muted); border-radius: 4px;">
                        <strong>Target Answer Span:</strong> "${item.target_sentence}"
                    </div>
                `;
                
                // Show Answer Toggle Action
                const showAnsBtn = itemDiv.querySelector(".show-ans-btn");
                const ansReveal = itemDiv.querySelector(".ans-reveal");
                showAnsBtn.addEventListener("click", () => {
                    ansReveal.classList.toggle("hidden");
                    // Highlight the target sentence in highlight viewer
                    selectSentence(item.sentence_idx);
                });
                
                quizItemsContainer.appendChild(itemDiv);
            });
            
            quizCard.classList.remove("hidden");
            
        } catch (e) {
            alert("Error generating quiz: " + e.message);
        } finally {
            quizBtn.disabled = false;
            quizBtn.querySelector("span").innerText = "Generate Quiz";
            quizBtn.querySelector("i").className = "fa-solid fa-graduation-cap";
        }
    });
});
