import React, { useEffect, useState } from "react";
// import { motion, useAnimation } from "framer-motion";
import { motion } from "framer-motion";
import "./FirstSection.css";

const FirstSection = () => {
    const [backgroundColor, setBackgroundColor] = useState("#4553f2");
    //   const controls = useAnimation();

    useEffect(() => {
        const handleScroll = () => {
            const scrollY = window.scrollY;
            if (scrollY < 300) setBackgroundColor("#4553f2");
            else if (scrollY < 600) setBackgroundColor("#171a3b");
            else setBackgroundColor("#171a3b");
        };

        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    const circleVariants = {
        initial: { scale: 1, opacity: 0.5, x: 0, y: 0 },
        left: {
          scale: [1.2, 1.4, 1.2],
          opacity: [0.5, 1, 0.5],
          x: [10, -60, 10], // Move left and back
          y: [-10, -60, -10], // Move down and back
          transition: {
            repeat: Infinity,
            repeatType: "mirror",
            duration: 10,
            ease: "easeInOut",
          },
        },
        right: {
          scale: [1.2, 1.4, 1.2],
          opacity: [0.5, 1, 0.5],
          x: [10, 60, 10], // Move right and back
          y: [-10, -60, -10], // Move up and back
          transition: {
            repeat: Infinity,
            repeatType: "mirror",
            duration: 10,
            ease: "easeInOut",
          },
        },
      };
      
      return (
        <div className="first-section" style={{ backgroundColor }}>
          <motion.div className="circles" initial="initial">
            {/* Left Group of Circles */}
            {[0, 1].map((i) => (
              <motion.div
                key={`left-${i}`}
                className="circle"
                variants={circleVariants}
                animate="left" // Correctly matches the key in `circleVariants`
              />
            ))}
      
            {/* Right Group of Circles */}
            {[0, 1].map((i) => (
              <motion.div
                key={`right-${i}`}
                className="circle"
                variants={circleVariants}
                animate="right" // Correctly matches the key in `circleVariants`
              />
            ))}
          </motion.div>

            <div className="content">
                <h1>Quick Tool to Generate Insights</h1>
                <h2>anaLIGHT</h2>
            </div>
        </div>
    );
};

export default FirstSection;
