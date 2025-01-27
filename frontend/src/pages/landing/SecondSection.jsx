import React from "react";
import {useNavigate} from "react-router-dom"
import "./SecondSection.css";

const SecondSection = () => {
    const navigate = useNavigate()

    const onGenerateInsights = (event) => {
        event.preventDefault(); // Prevent default form submission behavior
        // Add any additional logic or validation here if needed
        navigate("/insights"); // Navigate to the Insights page
      };

    return (
        <div className="second-section">
            {/* <header className="header">
        <div className="logo">PMX</div>
        <div className="say-hello">Say Hello</div>
      </header> */}

            <div className="content">
                <h1>Are You Ready?</h1>
                <p>
                    anaLIGHT is your personal reading assistant, generating insights and visualisations for the article you want to read. Drop us an article below and start generating insights!
                </p>

                <form className="article-form" onSubmit={onGenerateInsights}>
                    {/* Article Title */}
                    <div className="form-group">
                        <label htmlFor="article-title">Article Title<span id="required-sign">*</span> :</label>
                        <input
                            type="text"
                            id="article-title"
                            name="articleTitle"
                            placeholder="Enter the title of the article"
                            required
                        />
                    </div>

                    {/* Article Link */}
                    <div className="form-group">
                        <label htmlFor="article-link">Article Link<span id="required-sign">*</span> :</label>
                        <input
                            type="url"
                            id="article-link"
                            name="articleLink"
                            placeholder="Enter the link to the article"
                            required
                        />
                    </div>

                    {/* Article Description */}
                    <div className="form-group">
                        <label htmlFor="article-description">Article Description <span id="optional">(optional)</span>:</label>
                        <textarea
                            id="article-description"
                            name="articleDescription"
                            placeholder="Add a brief description"
                        />
                    </div>

                    {/* Generate Insights Button */}
                    <button type="submit" className="generate-button">
                        Generate Insights
                    </button>
                </form>
            </div>
        </div>
    );
};

export default SecondSection;
