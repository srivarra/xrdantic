import React from "react";
import XarrayLogo from "../../components/XarrayLogo.jsx";
import {
    CoordinatesIcon,
    DataArrayIcon,
    DatasetIcon,
    DataTreeIcon,
} from "../../components/DataTypeIcons.jsx";

const XrdanticBanner = ({
    revision = "0.0.1",
    width = 1280,
    height = 640,
    githubUrl = "srivarra/xrdantic",
    title = "XRDANTIC",
    subtitle = "Pydantic Scaffolding for Xarray",
    colors = {},
}) => {
    const defaultColors = {
        primary: "#4A90E2",
        secondary: "#2C5282",
        accent: "#6BE8E8",
        darkBlue: "#216C89",
        mediumBlue: "#4993AA",
        lightBlue: "#0F4565",
        cyan: "#6BE8E8",
        lightCyan: "#9DEEF4",
        teal: "#4ACFDD",
        orange: "#E38017",
        green: "#16AFB5",
        ...colors,
    };

    return (
        <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            xmlns="http://www.w3.org/2000/svg"
        >
            {/* Transparent background */}
            <rect
                width={width}
                height={height}
                fill="transparent"
                rx="19.2"
                ry="19.2"
            />

            {/* Dashed border */}
            <rect
                x="16"
                y="16"
                width={width - 32}
                height={height - 32}
                fill="none"
                stroke={defaultColors.primary}
                strokeWidth="3.2"
                strokeDasharray="12.8,6.4"
                opacity="0.7"
                rx="16"
                ry="16"
            />

            {/* Definitions for patterns */}
            <defs>
                <pattern
                    id="grid"
                    width="32"
                    height="32"
                    patternUnits="userSpaceOnUse"
                >
                    <path
                        d="M 32 0 L 0 0 0 32"
                        fill="none"
                        stroke={defaultColors.primary}
                        strokeWidth="0.8"
                        opacity="0.3"
                    />
                </pattern>

                <pattern
                    id="hatch-dark-blue"
                    patternUnits="userSpaceOnUse"
                    width="12.8"
                    height="12.8"
                >
                    <path
                        d="M0,12.8 L12.8,0"
                        stroke={defaultColors.darkBlue}
                        strokeWidth="1.28"
                        opacity="0.3"
                    />
                </pattern>

                <pattern
                    id="hatch-medium-blue"
                    patternUnits="userSpaceOnUse"
                    width="9.6"
                    height="9.6"
                >
                    <path
                        d="M0,9.6 L9.6,0"
                        stroke={defaultColors.mediumBlue}
                        strokeWidth="0.96"
                        opacity="0.25"
                    />
                </pattern>

                <pattern
                    id="hatch-light-blue"
                    patternUnits="userSpaceOnUse"
                    width="16"
                    height="16"
                >
                    <path
                        d="M0,16 L16,0"
                        stroke={defaultColors.lightBlue}
                        strokeWidth="1.6"
                        opacity="0.2"
                    />
                </pattern>

                <pattern
                    id="hatch-cyan"
                    patternUnits="userSpaceOnUse"
                    width="11.2"
                    height="11.2"
                >
                    <path
                        d="M0,0 L11.2,11.2"
                        stroke={defaultColors.cyan}
                        strokeWidth="1.12"
                        opacity="0.3"
                    />
                </pattern>

                <pattern
                    id="hatch-light-cyan"
                    patternUnits="userSpaceOnUse"
                    width="8"
                    height="8"
                >
                    <path
                        d="M0,0 L8,8"
                        stroke={defaultColors.lightCyan}
                        strokeWidth="0.8"
                        opacity="0.25"
                    />
                </pattern>

                <pattern
                    id="hatch-teal"
                    patternUnits="userSpaceOnUse"
                    width="14.4"
                    height="14.4"
                >
                    <path
                        d="M0,0 L14.4,14.4"
                        stroke={defaultColors.teal}
                        strokeWidth="1.28"
                        opacity="0.2"
                    />
                </pattern>

                <pattern
                    id="hatch-orange"
                    patternUnits="userSpaceOnUse"
                    width="9.6"
                    height="9.6"
                >
                    <path
                        d="M0,0 L9.6,9.6 M0,9.6 L9.6,0"
                        stroke={defaultColors.orange}
                        strokeWidth="0.96"
                        opacity="0.3"
                    />
                </pattern>

                <pattern
                    id="hatch-green"
                    patternUnits="userSpaceOnUse"
                    width="12.8"
                    height="12.8"
                >
                    <path
                        d="M0,0 L12.8,12.8"
                        stroke={defaultColors.green}
                        strokeWidth="1.12"
                        opacity="0.25"
                    />
                </pattern>
            </defs>

            {/* Grid background */}
            <rect
                width={width}
                height={height}
                fill="url(#grid)"
                rx="19.2"
                ry="19.2"
            />

            {/* Corner markers */}
            <g
                stroke={defaultColors.primary}
                strokeWidth="3.2"
                fill="none"
                opacity="0.8"
                strokeLinecap="round"
            >
                <path d="M 40 40 L 72 40 M 40 40 L 40 72" />
                <path
                    d={`M ${width - 40} 40 L ${width - 72} 40 M ${width - 40} 40 L ${width - 40} 72`}
                />
                <path
                    d={`M 40 ${height - 40} L 72 ${height - 40} M 40 ${height - 40} L 40 ${height - 72}`}
                />
                <path
                    d={`M ${width - 40} ${height - 40} L ${width - 72} ${height - 40} M ${width - 40} ${height - 40} L ${width - 40} ${height - 72}`}
                />
            </g>

            {/* GitHub URL */}
            <text
                x="80"
                y="56"
                fontFamily="'Geist Mono', 'Courier New', monospace"
                fontSize="16"
                fill={defaultColors.primary}
                textAnchor="start"
            >
                {githubUrl}
            </text>

            {/* Revision */}
            <text
                x={width - 80}
                y={height - 56}
                fontFamily="'Geist Mono', 'Courier New', monospace"
                fontSize="16"
                fill={defaultColors.primary}
                textAnchor="end"
            >
                REV: {revision}
            </text>

            {/* Xarray logo */}
            <XarrayLogo colors={defaultColors} />

            {/* Title */}
            <text
                x={width / 2}
                y="160"
                fontFamily="'Geist Mono', 'Courier New', monospace"
                fontSize="64"
                fontWeight="bold"
                fill={defaultColors.secondary}
                textAnchor="middle"
            >
                {title}
            </text>

            {/* Subtitle */}
            <text
                x={width / 2}
                y="200"
                fontFamily="'Geist Mono', 'Courier New', monospace"
                fontSize="28"
                fill={defaultColors.primary}
                textAnchor="middle"
            >
                {subtitle}
            </text>

            {/* Data type icons grid */}
            <g transform={`translate(${width * 0.65}, 240) scale(1.2)`}>
                {/* Grid background */}
                <rect
                    x="-230"
                    y="-20"
                    width="500"
                    height="280"
                    fill="none"
                    stroke={defaultColors.primary}
                    strokeWidth="1.6"
                    strokeDasharray="4.8,4.8"
                    opacity="0.3"
                    rx="6.4"
                />

                {/* Data type icons would go here */}
                {/* Top Left: Coordinates */}
                <CoordinatesIcon
                    colors={defaultColors}
                    transform="translate(-140, 50) translate(0, -18)"
                />
                {/* Top Right: DataArray */}
                <DataArrayIcon
                    colors={defaultColors}
                    transform="translate(140, 50) translate(0, -24)"
                />
                {/* Bottom Left: Dataset */}
                <DatasetIcon
                    colors={defaultColors}
                    transform="translate(-140, 170) translate(0, -30)"
                />
                {/* Bottom Right: DataTree */}
                <DataTreeIcon
                    colors={defaultColors}
                    transform="translate(140, 170) translate(0, -30)"
                />
            </g>
        </svg>
    );
};

export default XrdanticBanner;
