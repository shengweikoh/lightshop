import React from "react";
import "./InsightsPage.css"; // External CSS file for styling

function InsightsPage() {
    const articleData = {
        articleTitle: "Main Article Title",
        articleLink: "https://fake-link.com/?title=xxx",
        articleDescription: "This is the main article description...",
        keyEntities: ["Entity 1", "Entity 2", "Entity 3"],
        relatedArticles: [
            {
                title: "Related Article 1",
                link: "https://related-link-1.com",
                description:
                    "This is the description of related article 1, truncated to 200 words.",
            },
            {
                title: "Related Article 2",
                link: "https://related-link-2.com",
                description:
                    "This is the description of related article 2, truncated to 200 words.",
            },
        ],
    };

    return (
        <div className="insights-page">
            <div className="article-header">
                <h1>Article Title</h1>
                {/* retrieve article link? */}
                <a
                    href="https://fake-link123.com/?title=xxx"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    https://fake-link123.com/?title=xxx
                </a>
            </div>

            {/* Article Description */}
            <div className="article-description">
                <h2>Article Description:</h2>
                <p>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin vitae
                    neque at justo ultrices condimentum.
                </p>
            </div>

            {/* Key Entities Identified */}
            <div className="key-entities">
                <h2>Key Entities Identified:</h2>
                <div className="entities-list">
                    <div className="entity">Entity 1</div>
                    <div className="entity">Entity 2</div>
                    <div className="entity">Entity 3</div>
                </div>
            </div>

            {/* Visualization */}
            <div className="visualization">
                <h2>Visualization:</h2>
                <div className="visualization-placeholder">
                    {/* Replace with your actual visualization component */}
                    <p>Graph or Visualization will be displayed here.</p>
                </div>
            </div>

            {/* Related Articles */}
            <div className="related-articles">
                <h2>Related Articles:</h2>
                {articleData.relatedArticles.map((related, index) => (
                    <div key={index} className="related-article">
                        <h3>{related.title}</h3>
                        <a
                            href={related.link}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            {related.link}
                        </a>
                        <p>
                            {related.description}
                        </p>
                    </div>
                ))}

            </div>
        </div>
    );
}

export default InsightsPage;
