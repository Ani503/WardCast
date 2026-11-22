---
name: Clinical Intelligence
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#424751'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#727783'
  outline-variant: '#c2c6d3'
  surface-tint: '#115eaf'
  primary: '#004485'
  on-primary: '#ffffff'
  primary-container: '#0b5cad'
  on-primary-container: '#c0d6ff'
  inverse-primary: '#a8c8ff'
  secondary: '#515f74'
  on-secondary: '#ffffff'
  secondary-container: '#d5e3fc'
  on-secondary-container: '#57657a'
  tertiary: '#723200'
  on-tertiary: '#ffffff'
  tertiary-container: '#974400'
  on-tertiary-container: '#ffcaad'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d5e3ff'
  primary-fixed-dim: '#a8c8ff'
  on-primary-fixed: '#001b3c'
  on-primary-fixed-variant: '#004689'
  secondary-fixed: '#d5e3fc'
  secondary-fixed-dim: '#b9c7df'
  on-secondary-fixed: '#0d1c2e'
  on-secondary-fixed-variant: '#3a485b'
  tertiary-fixed: '#ffdbc9'
  tertiary-fixed-dim: '#ffb68d'
  on-tertiary-fixed: '#321200'
  on-tertiary-fixed-variant: '#763300'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  key-number:
    fontFamily: Inter
    fontSize: 64px
    fontWeight: '700'
    lineHeight: 72px
    letterSpacing: -0.02em
  key-number-mobile:
    fontFamily: Inter
    fontSize: 44px
    fontWeight: '700'
    lineHeight: 52px
    letterSpacing: -0.02em
  page-question:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  page-question-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.01em
  body-default:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  card-label:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 24px
  footer-auxiliary:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-compact:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.5rem
  margin: 2.5rem
  space-xs: 0.5rem
  space-sm: 1rem
  space-md: 1.5rem
  space-lg: 2rem
  space-xl: 2.5rem
---

## Brand & Style
The design system establishes a clean clinical minimalist language designed for high-stress healthcare environments, inpatient forecasting, and operational ward management. 

- **Audience:** Clinicians, charge nurses, hospital administrators, and bed-management coordinators requiring instant legibility without visual distraction.
- **Personality:** Decisive, disciplined, objective, and calm.
- **Emotional Response:** Confidence, clarity under pressure, and zero cognitive fatigue.
- **Style Direction:** Pure clinical minimalism with crisp boundaries, flat surfaces, zero gradients, zero elevation shadows, and an uncompromising adherence to structured hierarchy.

## Colors
The palette relies on strict functional categorization:

- **Background:** `#F8FAFC` provides a sterile, glare-reducing canvas for long shifts.
- **Main Blue (`#0B5CAD`):** The primary driver for key actions, brand anchor points, and active UI states.
- **Primary Text (`#0F172A`):** Maximum contrast slate navy for high-priority numbers and core data values.
- **Muted Text (`#475569`):** Subordinate metadata, labels, and secondary context.
- **Border Neutral (`#E2E8F0`):** Used universally for all structural card boundaries and field outlines.
- **Clinical Severity Scale:**
  - **Critical (`#C62828`):** Immediate attention, clinical deterioration, or ward capacity limits.
  - **Warning (`#B26A00`):** Projected bottlenecks, pending labs, or borderline capacity.
  - **Normal (`#2E7D32`):** Stable metrics, discharged beds, and nominal flow.

No color gradients or visual filters are permitted; all color fills must be flat solid values.

## Typography
All typography is standardized on **Inter** to ensure geometric clarity and optical balance across medical data tables, metrics, and labels:

- **Key Numbers (`64px` Bold):** Reserved for primary operational telemetry, bed census counts, and critical percentages.
- **Page Question / Hero Header (`32px` Bold):** Sets primary ward decisions and page-level queries.
- **Body Text (`18px` Regular):** Standard reading level for clinical notes, descriptions, and instructions.
- **Card Labels (`16px` Medium):** Muted metadata, card overlines, and metric labels.
- **Strict Size Floor (`14px`):** No element, metadata tag, timestamp, or footer text may drop below 14px under any circumstance.

## Layout & Spacing
The layout adheres strictly to an 8px base coordinate system framed within a standardized viewport strategy:

- **Desktop Standard:** 1440x900 viewport target with a 1200px maximum width centered main content column.
- **Top Navigation Bar:** Exactly 64px fixed height, pinned with a solid bottom border.
- **Page Margin:** 40px outer padding (`2.5rem`) on desktop canvas boundaries.
- **Card Metrics:** Internal card padding is uniform at 24px (`1.5rem`).
- **Gaps & Rhythm:** Uniform 24px (`1.5rem`) gap separating layout grid columns and stacked cards.
- **Responsive Adaptations:** 
  - On tablet viewports (<1024px), page margins scale down to 24px, and column grids reflow from 3-4 columns to 2 columns.
  - On mobile devices (<768px), page margins reduce to 16px, cards collapse into a single-column stack, and key metric sizes adopt mobile tokens to prevent word wrapping.

## Elevation & Depth
In accordance with clean clinical minimalism:
- **No Drop Shadows:** Box-shadow and drop-shadow properties are strictly forbidden across all component tiers.
- **Structural Boundaries:** Visual separation is driven entirely through contrast between the `#F8FAFC` page background and `#FFFFFF` solid card surfaces, defined by an explicit `1px solid #E2E8F0` border.
- **Layering:** Modals and temporary flyouts sit directly on flat, semi-opaque backdrops (`#0F172A` with 30% alpha), relying on sharp 1px borders rather than diffuse ambient shadows.

## Shapes
- **Card Radii:** Exactly 12px (`ROUND_TWELVE`) applied uniformly to all primary cards, content panels, and modal containers.
- **Inner Controls:** Buttons, inputs, and state chips use an 8px radius to nest symmetrically within 12px parent containers.
- **Structural Lines:** All dividing lines, separators, and perimeter outlines are calibrated to exact 1px widths.

## Components

### Cards & Panels
- **Surface:** Solid `#FFFFFF` fill with `12px` corner radius.
- **Border:** `1px solid #E2E8F0`.
- **Padding:** Exactly `24px` internal padding.
- **Content Hierarchy:** Overline card label at `16px` medium in `#475569`, followed by the focal metric (`64px` bold in `#0F172A` or severity token), with auxiliary trends rendered at `14px`.

### Buttons
- **Primary:** Solid `#0B5CAD` background, `#FFFFFF` text (Inter 16px medium), 8px border radius, 0px to 12px horizontal padding, 44px total target height.
- **Secondary / Outlined:** `#FFFFFF` background, `1px solid #E2E8F0` border, `#0F172A` text. Hover state switches border to `#0B5CAD` without shadow.
- **Destructive / Urgent:** Flat `#C62828` fill with `#FFFFFF` text.

### Inputs & Form Elements
- **Height:** 44px fixed height with `14px` internal padding.
- **Border:** `1px solid #E2E8F0`, transitioning to `1px solid #0B5CAD` on focus (no glow rings).
- **Text:** `16px` regular Inter in `#0F172A` with placeholder text in `#475569`.
- **Background:** Flat `#FFFFFF`.

### Chips & Status Badges
- **Shape:** 6px to 8px border radius with a minimum height of 28px.
- **Severity Implementation:**
  - *Critical:* `#C62828` foreground, solid `#FDE8E8` background (or 1px solid `#C62828` with `#FFFFFF` fill).
  - *Warning:* `#B26A00` foreground, solid `#FEF3C7` background.
  - *Normal:* `#2E7D32` foreground, solid `#DCFCE7` background.
- **Typography:** `14px` medium, uppercase or title case.

### Lists & Tables
- **Rows:** Minimum height of 52px to accommodate high clinical touch accuracy.
- **Separators:** Bottom border `1px solid #E2E8F0`. No alternate alternating row fills; rows maintain flat `#FFFFFF`.
- **Text:** Cell values at `16px` or `18px`, aligned left; metric values aligned right.

### Top Navigation Bar
- **Dimensions:** Fixed `64px` height spanning the viewport.
- **Styling:** `#FFFFFF` background, `1px solid #E2E8F0` bottom border, 40px horizontal padding aligning precisely with the 1200px max-width content container.